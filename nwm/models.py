# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import math
from collections.abc import Mapping
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from motion_condition import (
    MotionConditionEncoder,
    RealToLatentAdapter,
    StateConditionedLatentController,
    make_motion_group,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ActionEmbedder(nn.Module):
    """
    Embeds action xy into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        hsize = hidden_size // 3
        self.x_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.y_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.angle_emb = TimestepEmbedder(
            hidden_size - 2 * hsize, frequency_embedding_size
        )

    def forward(self, xya):
        return torch.cat(
            [
                self.x_emb(xya[..., 0:1]),
                self.y_emb(xya[..., 1:2]),
                self.angle_emb(xya[..., 2:3]),
            ],
            dim=-1,
        )


#################################################################################
#                                 Core CDiT Model                                #
#################################################################################


class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            add_bias_kv=True,
            bias=True,
            batch_first=True,
            **block_kwargs,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, c, x_cond):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca_xcond,
            scale_ca_xcond,
            shift_ca_x,
            scale_ca_x,
            gate_ca_x,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(11, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = (
            x
            + gate_ca_x.unsqueeze(1)
            * self.cttn(
                query=modulate(self.norm2(x), shift_ca_x, scale_ca_x),
                key=x_cond_norm,
                value=x_cond_norm,
                need_weights=False,
            )[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        motion_condition=None,
        training_stage="legacy",
        action_mode=None,
        proxy_relative_time_mode="always",
        finetune=None,
    ):
        super().__init__()
        self.context_size = context_size
        self.hidden_size = int(hidden_size)
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.training_stage = str(training_stage or "legacy")
        self.action_mode = str(action_mode or "real")
        self.proxy_relative_time_mode = str(
            proxy_relative_time_mode or "always"
        ).strip().lower()
        if self.proxy_relative_time_mode not in {"always", "fallback"}:
            raise ValueError(
                "proxy_relative_time_mode must be 'always' or 'fallback', got "
                f"{self.proxy_relative_time_mode!r}"
            )
        raw_finetune_scheme = (
            str((finetune or {}).get("scheme", "reset"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        self.finetune_scheme = {
            "align": "embedding_align",
            "alignment": "embedding_align",
            "embedding_alignment": "embedding_align",
            "d": "state_conditioned_controller",
            "state_controller": "state_conditioned_controller",
            "latent_state_controller": "state_conditioned_controller",
        }.get(raw_finetune_scheme, raw_finetune_scheme)
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.motion_condition_enabled = bool(
            motion_condition is not None
            and motion_condition.get("enabled", False)
        )
        if self.motion_condition_enabled:
            self.motion_condition_encoder = MotionConditionEncoder(
                hidden_size, motion_condition
            )
            if (
                self.training_stage == "real_finetune"
                and self.finetune_scheme == "real_to_latent"
            ):
                real_dim = self.motion_condition_encoder.input_dims.get("real")
                latent_dim = self.motion_condition_encoder.input_dims.get("latent")
                if real_dim is None or latent_dim is None:
                    raise ValueError(
                        "real_to_latent requires both real and latent motion adapters"
                    )
                mapper_hidden = int(
                    (finetune or {}).get("real_to_latent_hidden_dim", hidden_size)
                )
                self.real_to_latent = RealToLatentAdapter(
                    real_dim, mapper_hidden, latent_dim
                )
            elif (
                self.training_stage == "real_finetune"
                and self.finetune_scheme == "state_conditioned_controller"
            ):
                real_dim = self.motion_condition_encoder.input_dims.get("real")
                latent_dim = self.motion_condition_encoder.input_dims.get("latent")
                if real_dim is None or latent_dim is None:
                    raise ValueError(
                        "state_conditioned_controller requires both real and "
                        "latent motion adapters"
                    )
                self.state_conditioned_controller = StateConditionedLatentController(
                    real_dim,
                    hidden_size,
                    latent_dim,
                    num_heads=int(
                        (finetune or {}).get("controller_num_heads", num_heads)
                    ),
                    num_state_blocks=int(
                        (finetune or {}).get("controller_state_blocks", 2)
                    ),
                    mlp_ratio=float(
                        (finetune or {}).get("controller_mlp_ratio", 4.0)
                    ),
                )
        else:
            # Preserve the original module and state-dict keys for old configs and
            # checkpoints. The discrete models also continue to use ActionEmbedder.
            self.y_embedder = ActionEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(self.context_size + 1, num_patches, hidden_size),
            requires_grad=True,
        )  # for context and for predicted frame
        self.blocks = nn.ModuleList(
            [
                CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Keep the exact legacy initialization when the new framework is off.
        if not self.motion_condition_enabled:
            nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

            nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

            nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _dense_action_condition(
        self,
        base_condition,
        motion_type,
        action,
        action_valid=None,
    ):
        """Add a dense action embedding, masking only after the encoder."""
        embeddings = self.motion_condition_encoder.encode(motion_type, action)
        if embeddings.shape[0] != base_condition.shape[0]:
            raise ValueError(
                "Action and condition batch sizes differ: "
                f"{embeddings.shape[0]} != {base_condition.shape[0]}"
            )
        if action_valid is not None:
            action_valid = torch.as_tensor(
                action_valid, device=embeddings.device, dtype=torch.bool
            )
            if action_valid.shape != (embeddings.shape[0],):
                raise ValueError(
                    f"action_valid must be [{embeddings.shape[0]}], got "
                    f"{tuple(action_valid.shape)}"
                )
            embeddings = embeddings * action_valid.to(embeddings.dtype).unsqueeze(-1)
        return base_condition + embeddings.to(base_condition.dtype), embeddings

    def _group_real_to_latent_condition(self, base_condition, motion):
        if motion is None or len(motion) == 0:
            return base_condition, None
        if not isinstance(motion, Mapping) or set(motion).difference({"real"}):
            raise ValueError(
                "real_to_latent inference accepts grouped real action only"
            )
        payload = motion.get("real")
        if payload is None:
            return base_condition, None
        indices = payload.get("indices")
        values = payload.get("values")
        if not isinstance(indices, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("motion['real'] requires tensor indices and values")
        indices = indices.to(device=base_condition.device, dtype=torch.int64)
        if indices.ndim != 1 or values.ndim != 2 or values.shape[0] != indices.numel():
            raise ValueError("Grouped real motion indices/values are misaligned")
        if bool(((indices < 0) | (indices >= base_condition.shape[0])).any()):
            raise ValueError("Grouped real motion indices are outside the model batch")
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError("Grouped real motion indices contain duplicates")
        normalized_real = self.motion_condition_encoder.normalize("real", values)
        predicted_latent = self.real_to_latent(normalized_real)
        latent_embedding = self.motion_condition_encoder.encode(
            "latent", predicted_latent
        ).to(base_condition.dtype)
        return base_condition.index_add(0, indices, latent_embedding), predicted_latent

    def _predict_state_conditioned_latent(self, state_tokens, real_action):
        normalized_real = self.motion_condition_encoder.normalize("real", real_action)
        return self.state_conditioned_controller(state_tokens, normalized_real)

    def _group_state_conditioned_controller(
        self,
        base_condition,
        motion,
        state_tokens,
    ):
        if motion is None or len(motion) == 0:
            return base_condition, None
        if not isinstance(motion, Mapping) or set(motion).difference({"real"}):
            raise ValueError(
                "state_conditioned_controller inference accepts grouped real "
                "action only"
            )
        payload = motion.get("real")
        if payload is None:
            return base_condition, None
        indices = payload.get("indices")
        values = payload.get("values")
        if not isinstance(indices, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("motion['real'] requires tensor indices and values")
        indices = indices.to(device=base_condition.device, dtype=torch.int64)
        if indices.ndim != 1 or values.ndim != 2 or values.shape[0] != indices.numel():
            raise ValueError("Grouped real motion indices/values are misaligned")
        if bool(((indices < 0) | (indices >= base_condition.shape[0])).any()):
            raise ValueError("Grouped real motion indices are outside the model batch")
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError("Grouped real motion indices contain duplicates")
        predicted_latent = self._predict_state_conditioned_latent(
            state_tokens.index_select(0, indices),
            values,
        )
        latent_embedding = self.motion_condition_encoder.encode(
            "latent", predicted_latent
        ).to(base_condition.dtype)
        return base_condition.index_add(0, indices, latent_embedding), predicted_latent

    def _alignment_embeddings(self, action, teacher_latent, teacher_valid=None):
        if action is None or teacher_latent is None:
            raise ValueError("Alignment requires real action and teacher_latent")
        real_embedding = self.motion_condition_encoder.encode("real", action)
        with torch.no_grad():
            target_embedding = self.motion_condition_encoder.encode(
                "latent", teacher_latent
            )
        if real_embedding.shape != target_embedding.shape:
            raise ValueError(
                "Alignment embedding shape mismatch: "
                f"{tuple(real_embedding.shape)} != {tuple(target_embedding.shape)}"
            )
        if teacher_valid is None:
            teacher_valid = torch.ones(
                real_embedding.shape[0], dtype=torch.bool, device=real_embedding.device
            )
        else:
            teacher_valid = torch.as_tensor(
                teacher_valid, dtype=torch.bool, device=real_embedding.device
            )
        if teacher_valid.shape != (real_embedding.shape[0],):
            raise ValueError(
                f"teacher_valid must be [{real_embedding.shape[0]}], got "
                f"{tuple(teacher_valid.shape)}"
            )
        return {
            "real_embedding": real_embedding,
            "target_embedding": target_embedding,
            "alignment_valid": teacher_valid,
        }

    def _proxy_relative_time_embedding(
        self,
        relative_time_embedding,
        *,
        mode,
        action,
        action_valid,
        motion,
    ):
        """Keep relative time only on rows without a usable Stage-1 proxy."""
        if not (
            self.training_stage == "proxy_pretrain"
            and self.proxy_relative_time_mode == "fallback"
            and mode == "latent"
        ):
            return relative_time_embedding

        use_proxy = torch.zeros(
            relative_time_embedding.shape[0],
            dtype=torch.bool,
            device=relative_time_embedding.device,
        )
        if action is not None:
            if action_valid is None:
                use_proxy.fill_(True)
            else:
                use_proxy = torch.as_tensor(
                    action_valid,
                    dtype=torch.bool,
                    device=relative_time_embedding.device,
                )
                if use_proxy.shape != (relative_time_embedding.shape[0],):
                    raise ValueError(
                        "action_valid must be "
                        f"[{relative_time_embedding.shape[0]}], got "
                        f"{tuple(use_proxy.shape)}"
                    )
        elif motion:
            payload = motion.get(mode)
            if payload is not None:
                if not isinstance(payload, Mapping):
                    raise TypeError(f"motion[{mode!r}] must be a mapping")
                indices = payload.get("indices")
                if not isinstance(indices, torch.Tensor):
                    raise TypeError(
                        f"motion[{mode!r}] requires tensor indices and values"
                    )
                indices = indices.to(
                    device=relative_time_embedding.device, dtype=torch.int64
                )
                if indices.ndim != 1:
                    raise ValueError(f"motion[{mode!r}].indices must be int64 [N]")
                if bool(
                    (
                        (indices < 0)
                        | (indices >= relative_time_embedding.shape[0])
                    ).any()
                ):
                    raise ValueError(
                        f"motion[{mode!r}].indices are outside the model batch"
                    )
                use_proxy[indices] = True

        # Mask the embedding, not rel_t itself: TimestepEmbedder(0) is learned
        # and generally nonzero because its MLP has biases.
        return relative_time_embedding * (~use_proxy).to(
            relative_time_embedding.dtype
        ).unsqueeze(-1)

    def _compute_condition(
        self,
        diffusion_embedding,
        relative_time_embedding,
        *,
        y=None,
        motion=None,
        action=None,
        action_valid=None,
        conditioning_mode=None,
        teacher_latent=None,
        teacher_valid=None,
        state_tokens=None,
    ):
        base_condition = diffusion_embedding + relative_time_embedding
        auxiliary = {}

        # Preserve the pre-existing path byte-for-byte for old configs and calls.
        if self.training_stage == "legacy" and conditioning_mode is None and action is None:
            if self.motion_condition_enabled:
                if motion is None and y is not None:
                    motion = make_motion_group("real", y)
                return self.motion_condition_encoder(base_condition, motion), auxiliary
            if y is None:
                raise ValueError("Legacy CDiT requires the real action argument y")
            return base_condition + self.y_embedder(y), auxiliary

        if not self.motion_condition_enabled:
            raise ValueError("Two-stage conditioning requires motion_condition.enabled=true")

        if conditioning_mode is not None:
            mode = str(conditioning_mode).lower()
        elif self.training_stage == "proxy_pretrain":
            mode = self.action_mode.lower()
        elif self.finetune_scheme == "real_to_latent":
            mode = "real_to_latent"
        elif self.finetune_scheme == "state_conditioned_controller":
            mode = "state_conditioned_controller"
        else:
            mode = "real"

        if mode == "none":
            return base_condition, auxiliary

        if (
            action is None
            and self.training_stage == "proxy_pretrain"
            and mode in {"geometry", "idm", "latent"}
            and motion is not None
        ):
            supplied_types = set(motion)
            incompatible = supplied_types.difference({mode})
            if incompatible:
                raise ValueError(
                    "Stage-1 proxy conditioning cannot consume a different action "
                    f"representation: action_mode={mode!r}, supplied="
                    f"{sorted(supplied_types)}. Run real-action inference from a "
                    "stage-2 checkpoint instead."
                )

        relative_time_embedding = self._proxy_relative_time_embedding(
            relative_time_embedding,
            mode=mode,
            action=action,
            action_valid=action_valid,
            motion=motion,
        )
        base_condition = diffusion_embedding + relative_time_embedding

        if mode == "real_to_latent":
            if action is None:
                condition, predicted_latent = self._group_real_to_latent_condition(
                    base_condition, motion
                )
            else:
                normalized_real = self.motion_condition_encoder.normalize("real", action)
                predicted_latent = self.real_to_latent(normalized_real)
                latent_embedding = self.motion_condition_encoder.encode(
                    "latent", predicted_latent
                )
                if action_valid is not None:
                    valid = torch.as_tensor(
                        action_valid, dtype=torch.bool, device=latent_embedding.device
                    )
                    if valid.shape != (latent_embedding.shape[0],):
                        raise ValueError(
                            f"action_valid must be [{latent_embedding.shape[0]}], got "
                            f"{tuple(valid.shape)}"
                        )
                    latent_embedding = latent_embedding * valid.to(
                        latent_embedding.dtype
                    ).unsqueeze(-1)
                condition = base_condition + latent_embedding.to(base_condition.dtype)
            if predicted_latent is not None:
                auxiliary["predicted_latent"] = predicted_latent
            return condition, auxiliary

        if mode == "state_conditioned_controller":
            if state_tokens is None:
                raise ValueError(
                    "state_conditioned_controller requires current-state tokens"
                )
            if action is None:
                condition, predicted_latent = (
                    self._group_state_conditioned_controller(
                        base_condition,
                        motion,
                        state_tokens,
                    )
                )
            else:
                predicted_latent = self._predict_state_conditioned_latent(
                    state_tokens,
                    action,
                )
                latent_embedding = self.motion_condition_encoder.encode(
                    "latent", predicted_latent
                )
                if action_valid is not None:
                    valid = torch.as_tensor(
                        action_valid,
                        dtype=torch.bool,
                        device=latent_embedding.device,
                    )
                    if valid.shape != (latent_embedding.shape[0],):
                        raise ValueError(
                            f"action_valid must be [{latent_embedding.shape[0]}], "
                            f"got {tuple(valid.shape)}"
                        )
                    latent_embedding = latent_embedding * valid.to(
                        latent_embedding.dtype
                    ).unsqueeze(-1)
                condition = base_condition + latent_embedding.to(base_condition.dtype)
            if predicted_latent is not None:
                auxiliary["predicted_latent"] = predicted_latent
            return condition, auxiliary

        if action is not None:
            condition, action_embedding = self._dense_action_condition(
                base_condition, mode, action, action_valid
            )
        else:
            condition = self.motion_condition_encoder(base_condition, motion)
            action_embedding = None

        if (
            self.finetune_scheme == "embedding_align"
            and teacher_latent is not None
        ):
            alignment = self._alignment_embeddings(
                action, teacher_latent, teacher_valid
            )
            if action_embedding is not None:
                # Reuse the exact embedding that conditioned the NWM.
                alignment["real_embedding"] = action_embedding
            auxiliary.update(alignment)
        return condition, auxiliary

    def forward(
        self,
        x,
        t,
        y=None,
        x_cond=None,
        rel_t=None,
        motion=None,
        *,
        action=None,
        action_valid=None,
        conditioning_mode=None,
        teacher_latent=None,
        teacher_valid=None,
        return_aux=False,
        alignment_only=False,
        state_controller_only=False,
    ):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: legacy real action tensor. New callers should pass grouped ``motion``.
        motion: optional per-type mapping with ``indices`` and ``values`` tensors.
        """
        if alignment_only:
            if self.finetune_scheme != "embedding_align":
                raise ValueError("alignment_only is valid only for embedding_align")
            return self._alignment_embeddings(action, teacher_latent, teacher_valid)

        if x_cond is None:
            raise ValueError("CDiT requires x_cond")
        actual_context_size = int(x_cond.shape[1])
        if actual_context_size != self.context_size:
            raise ValueError(
                "CDiT context length does not match its positional embeddings: "
                f"received {actual_context_size}, configured {self.context_size}. "
                "Evaluation and planning context sizes must equal the training "
                "dataset.context_size recorded by the checkpoint."
            )

        if state_controller_only:
            if self.finetune_scheme != "state_conditioned_controller":
                raise ValueError(
                    "state_controller_only is valid only for "
                    "state_conditioned_controller"
                )
            if action is None:
                raise ValueError("state_controller_only requires a dense real action")
            state_tokens = (
                self.x_embedder(x_cond[:, -1])
                + self.pos_embed[self.context_size - 1]
            )
            predicted_latent = self._predict_state_conditioned_latent(
                state_tokens,
                action,
            )
            return {"predicted_latent": predicted_latent}

        x = self.x_embedder(x) + self.pos_embed[self.context_size :]
        x_cond_by_frame = (
            self.x_embedder(x_cond.flatten(0, 1)).unflatten(
                0, (x_cond.shape[0], x_cond.shape[1])
            )
            + self.pos_embed[: self.context_size]
        )
        state_tokens = x_cond_by_frame[:, -1]
        # (N, context * patches, D)
        x_cond = x_cond_by_frame.flatten(1, 2)
        t = self.t_embedder(t[..., None])
        time_emb = self.time_embedder(rel_t[..., None])
        c, auxiliary = self._compute_condition(
            t,
            time_emb,
            y=y,
            motion=motion,
            action=action,
            action_valid=action_valid,
            conditioning_mode=conditioning_mode,
            teacher_latent=teacher_latent,
            teacher_valid=teacher_valid,
            state_tokens=state_tokens,
        )

        for block in self.blocks:
            x = block(x, c, x_cond)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return (x, auxiliary) if return_aux else x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                  #
#################################################################################


def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


CDiT_models = {
    "CDiT-XL/2": CDiT_XL_2,
    "CDiT-L/2": CDiT_L_2,
    "CDiT-B/2": CDiT_B_2,
    "CDiT-S/2": CDiT_S_2,
}

"""DINOv2 CLS cosine distance on the same 224px crop used by navigation."""
import torch


class DinoMatcher:
    def __init__(self, model_path, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval().requires_grad_(False).to(device)
        if self.model.config.model_type != 'dinov2':
            raise ValueError('Matching model must be DINOv2')
        self.mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]

    @torch.inference_mode()
    def encode(self, rgb):
        features = self.model(pixel_values=(rgb - self.mean) / self.std).last_hidden_state[:, 0].float()
        if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
            raise RuntimeError('Invalid DINOv2 features')
        return torch.nn.functional.normalize(features, dim=-1)

    @staticmethod
    def distances(features, targets):
        return (1 - features @ targets.T).clamp(0, 2)

"""Lazy single-GPU OpenNWM inference shared by the CLI and HTTP worker."""
import hashlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scenes import REPO, source_path
from trajectory import FPS, FRAMES, PATH_UNITS, path_to_actions, actions_to_endpoints


def public_manifest(manifest):
    """Preserve reproducibility fields without publishing machine-local paths."""
    result = copy.deepcopy(manifest)
    result["source"].pop("path", None)
    result["checkpoint"].pop("path", None)
    return result


class Predictor:
    def __init__(self, run_dir, steps=250, mode=None):
        self.run_dir = Path(run_dir).resolve()
        self.steps = int(steps)
        self.loaded = False
        self.mode = mode

    def load(self):
        if self.loaded:
            return
        sys.path.insert(0, str(REPO / "nwm"))
        import torch
        from hydra_utils import load_experiment_config
        from train_utils import setup_model, setup_tokenizer, setup_diffusion
        from omegaconf import open_dict
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for live OpenNWM inference.")
        self.config = load_experiment_config(str(self.run_dir))
        if self.config is None:
            raise RuntimeError("Missing experiment configuration.")
        with open_dict(self.config):
            self.config.model.diffusion.eval_timestep_respacing = self.steps
        self.device = torch.device("cuda:0")
        print(f"Loading OpenNWM on {torch.cuda.get_device_name(self.device)}", flush=True)
        vae = setup_tokenizer(self.config, self.device).eval()
        # Deterministic VAE posterior means make the same drawn path and seed
        # reproducible in both single-request and batched preset generation.
        vae.encode = lambda images: vae.vae.encode(images).latent_dist.mode().mul_(vae.scaling_factor)
        model = setup_model(self.config, self.device)
        checkpoint = self.run_dir / "checkpoints/latest.pth.tar"
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(weights["ema"], strict=True)
        del weights
        model.eval()
        self.models = (model, setup_diffusion(self.config, True, self.device), vae)
        stat = checkpoint.stat()
        self.checkpoint = {"id": "joint_0100000", "path": str(checkpoint.resolve()), "bytes": stat.st_size, "mtimeNs": stat.st_mtime_ns}
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint["sha256"] = digest.hexdigest()
        self.loaded = True
        print("Model ready", flush=True)

    def predict(self, scene, requests, output_dirs, progress=None):
        """Generate a small batch; requests contain points and seed."""
        import torch
        from PIL import Image
        self.load()
        from isolated_nwm_infer import model_forward_wrapper
        from misc import get_transform, get_normalize
        started = time.monotonic()
        torch.set_num_threads(4)
        source = source_path(scene)
        actions = [path_to_actions(request["points"]) for request in requests]
        mode = self.mode or scene.get("predictionMode", "autoregressive")
        direct = mode == "direct"
        model_actions = [actions_to_endpoints(values) for values in actions] if direct else actions
        transform = get_transform(self.config.dataset.image_size, self.config.dataset.mean, self.config.dataset.std)
        with Image.open(source) as image:
            initial = transform(image.convert("RGB"))
        context = initial[None, None].repeat(len(requests), self.config.dataset.context_size, 1, 1, 1).to(self.device)
        controls = torch.tensor(model_actions, dtype=torch.float32, device=self.device)
        initial_pixels = ((initial * .5 + .5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        for directory in output_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            Image.fromarray(initial_pixels).save(Path(directory) / "initial.webp", quality=95)
        # A batch shares diffusion noise to isolate the effect of its paths.
        seeds = {request["seed"] for request in requests}
        if len(seeds) != 1:
            raise ValueError("Batched requests must share a seed.")
        normalize = get_normalize(self.config.dataset.mean, self.config.dataset.std)
        with torch.inference_mode():
            for index in range(FRAMES):
                pixels = model_forward_wrapper(self.models, context, controls[:, index:index + 1], index + 1 if direct else 1,
                    int(self.config.model.generator.input_size), self.device,
                    num_cond=int(self.config.dataset.context_size), motion_type="real",
                    sample_keys=[0] * len(requests), noise_seed=requests[0]["seed"],
                    noise_stream=f"playground/{scene['id']}/direct" if direct else f"playground/{scene['id']}/step={index}")
                for batch_index, directory in enumerate(output_dirs):
                    array = (pixels[batch_index].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
                    Image.fromarray(array).save(Path(directory) / f"{index:03d}.png")
                if not direct:
                    context = torch.cat((context[:, 1:], normalize(pixels).unsqueeze(1)), dim=1)
                if progress:
                    progress((index + 1) / FRAMES * .94)
                print(f"{scene['id']}: frame {index + 1}/{FRAMES}, batch={len(requests)}, elapsed={time.monotonic() - started:.1f}s", flush=True)
        manifests = []
        for request, controls_list, directory in zip(requests, actions, output_dirs):
            directory = Path(directory)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(directory / "%03d.png"), "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory / "prediction.mp4")], check=True)
            with Image.open(directory / "015.png") as image:
                image.save(directory / "poster.webp", quality=92)
            manifest = {"schemaVersion": 1, "sceneId": scene["id"], "source": {"dataset": scene["dataset"], "trajectory": scene["trajectory"], "frame": scene["frame"], "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                "checkpoint": self.checkpoint, "seed": request["seed"], "points": request["points"], "actions": controls_list,
                "mode": mode, "modelConditions": actions_to_endpoints(controls_list) if direct else controls_list,
                "fps": FPS, "frames": FRAMES, "duration": FRAMES / FPS, "diffusionSteps": self.steps,
                "protocol": "Single initial RGB repeated to four context frames; deterministic VAE posterior means; DDPM; real planar action adapter. " + ("Fixed initial context; direct prediction of each future endpoint and time; shared diffusion noise across horizons." if direct else "Autoregressive RGB re-encoding."),
                "canvasUnitsToWaypointUnits": PATH_UNITS, "waypointSpacingMeters": scene.get("waypointSpacing"),
                "simulatedSource": scene.get("simulated", False),
                "planetaryTimingNote": "Four seconds is a model-step convention, not the irregular mission capture interval." if scene.get("planetary") else None,
                "elapsedBatchSeconds": time.monotonic() - started}
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            manifests.append(manifest)
        if progress:
            progress(1.)
        return manifests

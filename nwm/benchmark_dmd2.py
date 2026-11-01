"""Paired small-sample quality and synchronized latency benchmark for DMD2."""
import argparse
import json
from pathlib import Path
import time

import lpips
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from distill_dmd2 import (check_data, condition_for, digest, load_config, load_weights,
                         seed_all, setup_tokenizer, setup_diffusion, take_batch, write_json)
from dmd2 import FourStepSchedule, TIMESTEPS
from omegaconf import open_dict


def summarize(rows):
    keys = ("psnr", "ssim", "lpips", "denoise_ms", "end_to_end_ms")
    return {method: {key: float(np.mean([r[key] for r in rows if r["method"] == method]))
                     for key in keys} for method in sorted({r["method"] for r in rows})}


def paired_differences(rows, seed):
    rng = np.random.default_rng(seed)
    base = {r["sample"]: r for r in rows if r["method"] == "teacher_250_ddpm"}
    distilled = {r["sample"]: r for r in rows if r["method"] == "student_4_dmd2"}
    out = {}
    for key in ("psnr", "ssim", "lpips"):
        values = np.array([distilled[i][key] - base[i][key] for i in sorted(base)])
        draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        out[key] = dict(student_minus_teacher=float(values.mean()),
                        bootstrap_95_percent_ci=np.quantile(draws, [.025, .975]).tolist())
    return out


@torch.no_grad()
def benchmark(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists() or (output / "samples.jsonl").exists():
        raise FileExistsError("Benchmark output exists; choose a new output directory")
    config = load_config(args.source_run)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    check_data(config, data)
    saved = torch.load(args.student, map_location="cpu", weights_only=False)
    teacher_hash = digest(args.checkpoint)
    if saved["metadata"]["teacher_sha256"] != teacher_hash:
        raise ValueError("Student belongs to a different teacher")
    if saved["metadata"]["data_sha256"] != digest(args.data):
        raise ValueError("Benchmark data differs from recorded training/holdout data")
    if saved["metadata"]["timesteps"] != list(TIMESTEPS):
        raise ValueError("Student sampler schedule mismatch")
    metadata = saved["metadata"]
    updates = saved["generator_updates"]
    del saved
    teacher = load_weights(config, args.checkpoint, "cuda").requires_grad_(False)
    student = load_weights(config, args.student, "cuda").requires_grad_(False)
    vae = setup_tokenizer(config, torch.device("cuda")).eval().requires_grad_(False)
    perceptual = lpips.LPIPS(net="alex").cuda().eval()
    full = setup_diffusion(config, False, torch.device("cuda"))
    schedule = FourStepSchedule(full.alphas_cumprod).cuda()
    with open_dict(config):
        config.model.diffusion.eval_timestep_respacing = 250
    ddpm250 = setup_diffusion(config, True, torch.device("cuda"))
    with open_dict(config):
        config.model.diffusion.eval_timestep_respacing = 4
    ddpm4 = setup_diffusion(config, True, torch.device("cuda"))
    methods = ["teacher_250_ddpm", "teacher_4_ddpm", "teacher_4_dmd2", "student_4_dmd2"]
    samples = data[args.split]
    manifest = json.loads((Path(args.data).parent / "manifest.json").read_text())
    records = [r for r in manifest if r["split"] == args.split]
    if len(records) != len(samples["target"]):
        raise ValueError("Manifest/sample count mismatch")
    if digest(Path(args.data).parent / "manifest.json") != data["metadata"]["manifest_sha256"]:
        raise ValueError("Manifest was modified after preparation")
    rows, preview_rows, preview_counts = [], [], {}
    rng = np.random.default_rng(args.seed)

    def run(method, batch, context_pixels, sample_seed):
        seed_all(sample_seed)
        model = student if method == "student_4_dmd2" else teacher
        noise = torch.randn_like(batch["target"])
        # Record full online VAE encoding + denoising + decoding, excluding disk I/O.
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_cond = vae.vae.encode(context_pixels.flatten(0, 1)).latent_dist.mode() * vae.scaling_factor
            online_batch = dict(batch, context=x_cond.unflatten(0, context_pixels.shape[:2]).float())
            condition = condition_for(model, online_batch, config)
            torch.cuda.synchronize()
            denoise_start = time.perf_counter()
            if method.endswith("ddpm"):
                sampler = ddpm250 if method == "teacher_250_ddpm" else ddpm4
                latent = sampler.p_sample_loop(model.forward, noise.shape, noise=noise,
                    model_kwargs=condition, clip_denoised=False, device="cuda", progress=False)
            else:
                latent = schedule.sample(model, noise, condition)
            torch.cuda.synchronize()
            denoise_end = time.perf_counter()
            pixels = vae.decode(latent, denormalize=True).float()
        torch.cuda.synchronize()
        ended = time.perf_counter()
        if not torch.isfinite(pixels).all():
            raise FloatingPointError(f"Non-finite decoded image: {method}")
        return pixels, (denoise_end-denoise_start)*1000, (ended-started)*1000

    first = take_batch(samples, [0])
    first_context = samples["context_pixels"][:1].cuda().float() / 127.5 - 1
    for method in methods:
        run(method, first, first_context, args.seed)
    print("Warmup complete", flush=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    for index, record in enumerate(records):
        batch = take_batch(samples, [index])
        context_pixels = samples["context_pixels"][index:index+1].cuda().float() / 127.5 - 1
        gt = samples["pixels"][index:index+1].cuda().float()/255
        generated = {}
        measurements = {m: [] for m in methods}
        # Interleave method order on every repeat to reduce warmup/thermal order bias.
        for repeat in range(args.repeats):
            for method in rng.permutation(methods):
                pixels, latency, total = run(method, batch, context_pixels, args.seed + index)
                measurements[method].append((latency, total))
                if repeat == 0:
                    generated[method] = pixels
        for method in methods:
            pixels = generated[method]
            mse = F_mse(pixels, gt)
            pred_np = pixels[0].permute(1, 2, 0).cpu().numpy()
            value = perceptual(pixels*2-1, gt*2-1).item()
            timing = np.asarray(measurements[method])
            row = dict(sample=index, dataset=record["dataset"], trajectory=record["trajectory"],
                       horizon_frames=record["horizon"], method=method,
                       psnr=float(-10*np.log10(max(mse, 1e-12))),
                       ssim=float(structural_similarity_index_measure(pixels, gt, data_range=1.)),
                       lpips=value, denoise_ms=float(np.median(timing[:, 0])),
                       end_to_end_ms=float(np.median(timing[:, 1])), repeats_ms=timing.tolist())
            rows.append(row)
            with (output / "samples.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            Image.fromarray((pred_np*255).round().astype(np.uint8)).save(image_dir/f"{index:03d}_{method}.png")
        images = [context_pixels[0, -1].add(1).div(2), gt[0]] + [generated[m][0] for m in methods]
        labels = ["Context", "Ground truth", "Teacher 250 DDPM", "Teacher 4 DDPM", "Teacher 4 DMD2", "Student 4 DMD2"]
        h, w = gt.shape[-2:]
        grid = Image.new("RGB", (w*len(images), h+40), "white")
        draw = ImageDraw.Draw(grid)
        for column, (pixels, label) in enumerate(zip(images, labels)):
            array = (pixels.permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8)
            grid.paste(Image.fromarray(array), (column*w, 40))
            draw.text((column*w+4, 4), label, fill="black")
        draw.text((4, 20), f"{record['dataset']} / sample {index} / +{record['horizon']} frames", fill="black")
        grid.save(image_dir / f"{index:03d}_comparison.png")
        if preview_counts.get(record["dataset"], 0) < 2:
            preview_rows.append(grid)
            preview_counts[record["dataset"]] = preview_counts.get(record["dataset"], 0) + 1
        print(f"sample {index+1}/{len(records)} done", flush=True)
    preview = Image.new("RGB", (preview_rows[0].width, sum(i.height for i in preview_rows)), "white")
    y = 0
    for grid in preview_rows:
        preview.paste(grid, (0, y))
        y += grid.height
    preview.save(output / "comparison.png")
    summary = summarize(rows)
    base, distilled = summary["teacher_250_ddpm"], summary["student_4_dmd2"]
    speedup = {k: base[k]/distilled[k] for k in ("denoise_ms", "end_to_end_ms")}
    results = dict(summary=summary, speedup=speedup, paired_quality=paired_differences(rows, args.seed),
        per_dataset={name: summarize([r for r in rows if r["dataset"] == name]) for name in sorted({r["dataset"] for r in rows})},
        sample_count=len(records), split=args.split, repeats=args.repeats, seed=args.seed,
        generator_updates=updates, critic_updates=updates*metadata["arguments"]["critic_ratio"],
        train_sample_count=len(data["train"]["target"]), teacher_sha256=teacher_hash,
        student_sha256=digest(args.student), device=torch.cuda.get_device_name(), torch_version=torch.__version__,
        timing="batch=1, BF16 autocast, one warmup per method; median of repeats per sample; CUDA synchronized; VAE encode/decode included only in end_to_end_ms; no I/O",
        limitations=["Short pilot, not a converged distillation result", "Small test set; no FID estimate",
                     "Fixed VAE posterior means", "Direct future frames only; no rollout/navigation success measurement"],
        training_metadata=metadata)
    write_json(output / "results.json", results)
    lines = [f"# DMD2 four-step pilot: {args.label}", "",
             f"{len(records)} held-out {args.split} windows; {len(data['train']['target'])} training windows; "
             f"{updates} generator / {results['critic_updates']} critic updates. Batch 1, {results['device']}.", "",
             "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Denoise ms ↓ | Encode + denoise + decode ms ↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        s = summary[method]
        lines.append(f"| {method} | {s['psnr']:.3f} | {s['ssim']:.4f} | {s['lpips']:.4f} | {s['denoise_ms']:.2f} | {s['end_to_end_ms']:.2f} |")
    lines += ["", f"Student speedup over teacher 250 DDPM: denoising {speedup['denoise_ms']:.2f}×; "
              f"including VAE {speedup['end_to_end_ms']:.2f}×.", "", "## Paired student minus teacher differences", ""]
    for key, value in results["paired_quality"].items():
        lines.append(f"- {key}: {value['student_minus_teacher']:.5f}; bootstrap 95% CI {value['bootstrap_95_percent_ci']}")
    lines += ["", "![Comparison](comparison.png)", "", "## Scope", ""] + [f"- {x}" for x in results["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(summary=summary, speedup=speedup), indent=2), flush=True)


def F_mse(a, b):
    return (a-b).square().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "checkpoint", "student", "data", "output", "label"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.set_num_threads(4)
    benchmark(args)

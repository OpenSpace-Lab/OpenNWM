#!/usr/bin/env python3
"""Score a planetary prediction manifest and make a self-contained local report.

The manifest at OUTPUT/manifest.json contains ``samples``. Each sample has id,
dataset, protocol, initial_path, prediction_paths, and optionally aligned gt_paths.
Paths must be absolute; saved images must already use the model's RGB transform.
Missing GT is never inferred from neighboring source images or synthetic controls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


METRICS = ("psnr", "ssim", "lpips")
MODEL = "OpenNWM"
BASELINE = "Copy initial"


def atomic_json(path, value):
    """Write strict JSON, using a string for mathematically infinite PSNR."""
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            if math.isnan(item):
                raise ValueError("NaN metric encountered")
            return "Infinity" if item > 0 else "-Infinity"
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        return item

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def protocol_name(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def load_tensor(paths):
    return torch.stack([
        torch.from_numpy(np.asarray(read_rgb(path), dtype=np.float32).copy()).permute(2, 0, 1) / 255
        for path in paths
    ])


def prediction_steps(sample):
    return sample.get("prediction_steps", list(range(1, len(sample["prediction_paths"]) + 1)))


def psnr_ssim(prediction, target):
    """Per-image PSNR and Gaussian SSIM for RGB tensors in [0, 1].

    SSIM matches the Wang et al. 11x11 Gaussian window (sigma 1.5), population
    covariance, data_range=1, channel mean, excluding the five-pixel border.
    """
    prediction, target = prediction.double(), target.double()
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    psnr = -10 * torch.log10(mse)
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM requires image width and height >= 11")
    axis = torch.arange(-5, 6, device=prediction.device, dtype=torch.float64)
    weights = torch.exp(-axis.square() / (2 * 1.5**2))
    weights /= weights.sum()
    kernel = (weights[:, None] * weights[None, :])[None, None].repeat(3, 1, 1, 1)
    moment_inputs = torch.cat((prediction, target, prediction.square(), target.square(), prediction * target))
    mx, my, exx, eyy, exy = F.conv2d(moment_inputs, kernel, groups=3).chunk(5)
    vx, vy, vxy = exx - mx.square(), eyy - my.square(), exy - mx * my
    ssim = ((2 * mx * my + .01**2) * (2 * vxy + .03**2) /
            ((mx.square() + my.square() + .01**2) * (vx + vy + .03**2)))
    return psnr.cpu().tolist(), ssim.mean(dim=(1, 2, 3)).cpu().tolist()


def validate_manifest(manifest):
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest.json must contain a nonempty samples list")
    identities = set()
    image_sizes = {}
    for sample in samples:
        for key in ("id", "dataset", "protocol", "initial_path", "prediction_paths"):
            if key not in sample:
                raise ValueError(f"Sample missing required field: {key}")
        identity = (str(sample["dataset"]), protocol_name(sample["protocol"]), str(sample["id"]))
        if identity in identities:
            raise ValueError(f"Duplicate sample identity: {identity}")
        identities.add(identity)
        predictions = sample["prediction_paths"]
        ground_truth = sample.get("gt_paths") or []
        if not isinstance(predictions, list) or not predictions:
            raise ValueError(f"Sample {identity} has no prediction paths")
        if not isinstance(ground_truth, list):
            raise ValueError(f"Sample {identity}: gt_paths must be a list")
        if ground_truth and len(ground_truth) != len(predictions):
            raise ValueError(f"Sample {identity}: GT and prediction counts differ")
        steps = prediction_steps(sample)
        if (not isinstance(steps, list) or len(steps) != len(predictions) or
                any(not isinstance(step, int) or step < 1 for step in steps) or
                steps != sorted(set(steps))):
            raise ValueError(f"Sample {identity}: prediction_steps must be positive, unique, increasing and aligned")
        paths = [sample["initial_path"], *predictions, *ground_truth]
        for path_string in paths:
            path = Path(path_string)
            if not path.is_absolute():
                raise ValueError(f"Expected an absolute image path: {path}")
            if path_string not in image_sizes:
                with Image.open(path) as image:
                    image_sizes[path_string] = image.size
                    image.verify()
        sizes = {image_sizes[path] for path in paths}
        if len(sizes) != 1 or min(next(iter(sizes))) < 11:
            raise ValueError(f"Sample {identity}: mismatched or too-small images: {sizes}")
    return samples


def metric_averages(rows):
    return {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}


def aggregate(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, values in groups.items():
        result.append({**dict(zip(fields, key)), "frame_count": len(values),
                       "sample_count": len({str(row["id"]) for row in values}),
                       **metric_averages(values)})
    return result


def score(samples, device, batch_size, model_label=MODEL):
    pairs = []
    for sample in samples:
        for step, prediction, gt in zip(prediction_steps(sample), sample["prediction_paths"], sample.get("gt_paths") or []):
            info = {"id": sample["id"], "dataset": str(sample["dataset"]),
                    "protocol": protocol_name(sample["protocol"]), "step": step}
            pairs.append(({**info, "model": model_label}, prediction, gt))
            pairs.append(({**info, "model": BASELINE}, sample["initial_path"], gt))
    if not pairs:
        return []
    import lpips
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    # Different sample resolutions are supported, without resizing saved data.
    groups = defaultdict(list)
    for pair in pairs:
        with Image.open(pair[1]) as image:
            groups[image.size].append(pair)
    rows = []
    with torch.inference_mode():
        for same_size in groups.values():
            for offset in range(0, len(same_size), batch_size):
                batch = same_size[offset:offset + batch_size]
                prediction = load_tensor([pair[1] for pair in batch]).to(device)
                target = load_tensor([pair[2] for pair in batch]).to(device)
                psnr, ssim = psnr_ssim(prediction, target)
                distances = perceptual(prediction * 2 - 1, target * 2 - 1).flatten().cpu().tolist()
                for pair, p, s, distance in zip(batch, psnr, ssim, distances):
                    rows.append({**pair[0], "psnr": p, "ssim": s, "lpips": float(distance)})
                print(f"Scored {len(rows)}/{len(pairs)} image pairs", flush=True)
    return rows


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def safe_name(sample):
    identity = f"{sample['dataset']}-{sample['id']}-{protocol_name(sample['protocol'])}"
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", identity)[:90].strip("-")
    return name + "-" + hashlib.sha256(identity.encode()).hexdigest()[:8]


def comparison_frame(sample, step, model_label=MODEL):
    initial = read_rgb(sample["initial_path"])
    columns = [("Initial", initial)]
    if sample.get("gt_paths"):
        columns.append(("GT", read_rgb(sample["gt_paths"][step])))
    columns.append((model_label, read_rgb(sample["prediction_paths"][step])))
    width, height = initial.size
    canvas = Image.new("RGB", (width * len(columns), height + 64), "#111820")
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(columns):
        canvas.paste(image, (column * width, 32))
        draw.text((column * width + 9, 7), label, fill="white", font=font(17))
    draw.text((9, height + 40), f"{sample['dataset']} / {sample['id']}  |  model step {prediction_steps(sample)[step]}",
              fill="#d3dbe6", font=font(14))
    # yuv420p encoders require even frame dimensions.
    if canvas.width % 2 or canvas.height % 2:
        padded = Image.new("RGB", (canvas.width + canvas.width % 2, canvas.height + canvas.height % 2), "#111820")
        padded.paste(canvas, (0, 0))
        canvas = padded
    return canvas


def make_videos(samples, root, fps, model_label=MODEL):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to generate comparison videos")
    directory = root / "comparisons"
    directory.mkdir(exist_ok=True)
    assets = {}
    for sample in samples:
        stem = safe_name(sample)
        target = directory / f"{stem}.mp4"
        temporary = directory / f".{stem}.tmp.mp4"
        first = comparison_frame(sample, 0, model_label)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-s", f"{first.width}x{first.height}", "-r", str(fps),
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for step in range(len(sample["prediction_paths"])):
                process.stdin.write((first if step == 0 else comparison_frame(sample, step, model_label)).tobytes())
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            status = process.wait()
            if status:
                raise RuntimeError(f"ffmpeg failed with status {status}: {error}")
        except BaseException:
            process.kill()
            process.wait()
            temporary.unlink(missing_ok=True)
            raise
        finally:
            process.stderr.close()
        os.replace(temporary, target)
        comparison_frame(sample, len(sample["prediction_paths"]) - 1, model_label).save(directory / f"{stem}.jpg", quality=92)
        assets[stem] = {"video": f"comparisons/{stem}.mp4", "poster": f"comparisons/{stem}.jpg"}
        print(f"Rendered {target.name}", flush=True)
    return assets


def representatives(samples, count):
    """Round-robin groups; evenly span each group's manifest order, never score."""
    groups = defaultdict(list)
    for sample in samples:
        groups[(str(sample["dataset"]), protocol_name(sample["protocol"]))].append(sample)
    quotas = {key: 0 for key in groups}
    remaining = min(max(count, 0), len(samples))
    while remaining:
        for key, group in groups.items():
            if remaining and quotas[key] < len(group):
                quotas[key] += 1
                remaining -= 1
    selections = {
        key: [group[index] for index in np.linspace(0, len(group) - 1, quotas[key]).round().astype(int)]
        for key, group in groups.items()
    }
    chosen = []
    while any(selections.values()):
        for group in selections.values():
            if group and len(chosen) < count:
                chosen.append(group.pop(0))
    return chosen


def make_overview(samples, path, max_samples=8, model_label=MODEL):
    selected = representatives(samples, max_samples)
    tile = 224
    row_height = tile + 66
    canvas = Image.new("RGB", (5 * tile, len(selected) * row_height), "#111820")
    draw = ImageDraw.Draw(canvas)
    for row, sample in enumerate(selected):
        top = row * row_height
        title = f"{sample['dataset']} / {sample['id']} | {protocol_name(sample['protocol'])}"
        draw.text((10, top + 6), title[:140], fill="white", font=font(15))
        steps = sorted({0, (len(sample["prediction_paths"]) - 1) // 2, len(sample["prediction_paths"]) - 1})
        columns = [("Initial / copy baseline", sample["initial_path"])]
        columns.append(("GT final", sample["gt_paths"][-1]) if sample.get("gt_paths") else ("No paired GT", None))
        columns.extend((f"{model_label} step {prediction_steps(sample)[step]}", sample["prediction_paths"][step]) for step in steps)
        for column, (label, image_path) in enumerate(columns):
            x = column * tile
            draw.text((x + 9, top + 33), label, fill="#d3dbe6", font=font(15))
            if image_path:
                canvas.paste(read_rgb(image_path).resize((tile, tile)), (x, top + 60))
    canvas.save(path, quality=94)
    return [safe_name(sample) for sample in selected]


def number(value):
    value = float(value)
    return "∞" if math.isinf(value) else f"{value:.4f}"


def metric_table(rows, fields):
    headings = {"dataset": "Dataset", "protocol": "Protocol", "id": "Sample", "step": "Step",
                "model": "Prediction", "frame_count": "Frames", "sample_count": "Samples",
                "psnr": "PSNR ↑ (dB)", "ssim": "SSIM ↑", "lpips": "LPIPS ↓"}
    columns = [*fields, "sample_count", "frame_count", *METRICS]
    output = ["<table><thead><tr>" + "".join(f"<th>{headings[field]}</th>" for field in columns) + "</tr></thead><tbody>"]
    for row in rows:
        output.append("<tr>" + "".join("<td>" + html.escape(number(row[field]) if field in METRICS else str(row[field])) + "</td>" for field in columns) + "</tr>")
    return "".join([*output, "</tbody></table>"])


def make_report(root, manifest, metrics, samples, assets, fps):
    metadata = {key: value for key, value in manifest.items() if key != "samples"}
    model_label = manifest.get("model_label", MODEL)
    datasets = " / ".join(dict.fromkeys(str(sample["dataset"]) for sample in samples))
    title = html.escape(f"{model_label} — {datasets} evaluation")
    parts = ["""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + title + """</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#e5edf6;background:#10161f;margin:24px auto;padding:0 20px;max-width:1240px}
h1,h2,h3{line-height:1.2}a{color:#a9cffd}img,video{max-width:100%;height:auto}video{background:#080b10;width:100%}
article{border:1px solid #3b4656;border-radius:8px;padding:16px;margin:24px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{border:1px solid #3b4656;padding:8px;text-align:left}
th{background:#223042}summary{cursor:pointer}p.note{color:#b5c5d8}td{max-width:380px;overflow-wrap:anywhere}</style>
<h1>""" + title + """</h1>""",
             f"<p>{len(samples)} samples; {metrics['scored_sample_count']} samples with paired GT. "
             f"Comparison videos display saved predictions at {fps:g} FPS. Direct-prediction targets can have unequal gaps; "
             "captions show their model steps. Playback is a display convention unless the source timing is calibrated.</p>",
             "<p class='note'>Only samples with supplied, aligned GT receive metrics. Copy initial repeats the exact initial image for every target. "
             "PSNR and SSIM are higher-is-better; LPIPS is lower-is-better. Means weight each evaluated image equally. "
             "No FID is reported for this small evaluation. Qualitative examples are evenly spaced through manifest order within each dataset/protocol; "
             "selection never uses scores.</p>",
             "<p><a href='manifest.json'>Input manifest</a> · <a href='metrics.json'>Metric details</a> · <a href='metrics.csv'>Per-frame CSV</a></p>",
             "<h2>Quantitative results</h2>"]
    if metrics["aggregates"]:
        parts.append("<div class='scroll'>" + metric_table(metrics["aggregates"], ["dataset", "protocol", "model"]) + "</div>")
        parts.append("<details><summary>Metrics by prediction step</summary><div class='scroll'>" +
                     metric_table(metrics["per_step"], ["dataset", "protocol", "model", "step"]) + "</div></details>")
    else:
        parts.append("<p>No samples have paired GT; this report contains qualitative predictions only.</p>")
    parts.extend(["<h2>Overview</h2><a href='overview.jpg'><img src='overview.jpg' alt='Representative initial, reference and predicted images'></a>",
                  "<h2>Checkpoint and protocol</h2><pre>" + html.escape(json.dumps(metadata, indent=2, ensure_ascii=False)) + "</pre>",
                  "<h2>All samples</h2>"])
    for sample in samples:
        asset = assets[safe_name(sample)]
        info = {key: value for key, value in sample.items() if key not in {"initial_path", "gt_paths", "prediction_paths"}}
        parts.extend(["<article><h3>" + html.escape(f"{sample['dataset']} / {sample['id']}") + "</h3>",
                      "<p>" + html.escape(protocol_name(sample["protocol"])) + "</p>",
                      f"<video controls preload='none' poster='{asset['poster']}' src='{asset['video']}'></video>"])
        if not sample.get("gt_paths"):
            parts.append("<p class='note'>No paired ground truth: generated sequence only; quantitative accuracy is not measured.</p>")
        parts.append("<details><summary>Sample provenance</summary><pre>" + html.escape(json.dumps(info, indent=2, ensure_ascii=False)) + "</pre></details></article>")
    parts.append("<details><summary>Metric definitions</summary><pre>" + html.escape(json.dumps(metrics["definitions"], indent=2)) + "</pre></details></html>\n")
    (root / "report.html").write_text("".join(parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="LPIPS/SSIM device, normally cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fps", type=float, default=4, help="Comparison-video playback FPS, not source acquisition rate")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1 or args.fps <= 0 or not math.isfinite(args.fps):
        parser.error("batch-size, threads and fps must be positive")
    torch.set_num_threads(args.threads)
    root = args.output_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = validate_manifest(manifest)
    model_label = manifest.get("model_label", MODEL)
    if not isinstance(model_label, str) or not model_label.strip() or model_label == BASELINE:
        raise ValueError("model_label must be a nonempty string distinct from the copy baseline")
    print(f"Validated {len(samples)} samples; scoring on {args.device}", flush=True)
    rows = score(samples, torch.device(args.device), args.batch_size, model_label)
    metrics = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "sample_count": len(samples),
        "scored_sample_count": sum(bool(sample.get("gt_paths")) for sample in samples),
        "unscored_samples": [{"id": sample["id"], "dataset": sample["dataset"], "protocol": sample["protocol"],
                              "reason": "No paired ground truth supplied"} for sample in samples if not sample.get("gt_paths")],
        "definitions": {
            "psnr": "-10 log10(RGB MSE), input range [0,1]; exact equality is JSON string Infinity",
            "ssim": "RGB mean, 11x11 Gaussian window sigma=1.5, valid window, population covariance, K1=.01, K2=.03, data_range=1",
            "lpips": "LPIPS AlexNet v0.1, pretrained weights, inputs RGB [-1,1]",
            "baseline": "Repeat each sample's exact initial_path image for every GT target",
            "aggregation": "Arithmetic mean of per-image metrics, identical GT pairs for model and baseline; protocols remain separate",
            "steps": "sample.prediction_steps when supplied, otherwise one-based indices into prediction_paths; physical timing comes only from the input manifest",
            "image_processing": "Images are converted to RGB without crop or resize for scoring; 224 px resize used only in overview",
        },
        "aggregates": aggregate(rows, ["dataset", "protocol", "model"]),
        "per_step": aggregate(rows, ["dataset", "protocol", "model", "step"]),
        "per_sample": aggregate(rows, ["dataset", "protocol", "id", "model"]),
        "frames": rows,
    }
    atomic_json(root / "metrics.json", metrics)
    with (root / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "protocol", "id", "model", "step", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    assets = make_videos(samples, root, args.fps, model_label)
    make_overview(samples, root / "overview.jpg", model_label=model_label)
    make_report(root, manifest, metrics, samples, assets, args.fps)
    print(f"Report: {root / 'report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

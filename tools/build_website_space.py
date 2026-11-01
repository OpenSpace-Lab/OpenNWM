"""Publish traceable planetary observations and genuine model predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "website"
OUT = SITE / "assets/space"
ARCHIVE = Path("/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/opennwm_pixel_corrected_20260924")
FINETUNED_PREFIX = "lusnar-finetuned-"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def trajectory_svg(points, path):
    xs, ys = zip(*points)
    extent = max(max(xs) - min(xs), max(ys) - min(ys), .1)
    mapped = [(32 + (x - min(xs)) / extent * 160, 192 - (y - min(ys)) / extent * 160) for x, y in points]
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in mapped)
    grid = "".join(f'<path d="M{v} 0V224M0 {v}H224"/>' for v in range(16, 224, 32))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 224 224" role="img" aria-label="Conditioning trajectory, up is forward"><rect width="224" height="224" fill="#111a23"/><g stroke="#28333f" stroke-width="1">{grid}</g><polyline points="{coords}" fill="none" stroke="#d3ed98" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="{mapped[0][0]:.1f}" cy="{mapped[0][1]:.1f}" r="6" fill="#fff"/><circle cx="{mapped[-1][0]:.1f}" cy="{mapped[-1][1]:.1f}" r="6" fill="#d3ed98"/></svg>\n')


def render_video(columns, target, temporary_root, labels=(), poster=False):
    """Encode all sixteen original prediction frames, with no interpolation."""
    if any(len(column) != 16 for column in columns):
        raise ValueError("Each website video must contain exactly 16 frames")
    font = ImageFont.truetype(str(SITE / "assets/fonts/Manrope.ttf"), 12)
    with tempfile.TemporaryDirectory(prefix="website-frames-", dir=temporary_root) as temporary:
        temporary = Path(temporary)
        for index, paths in enumerate(zip(*columns)):
            header = 28 if labels else 0
            frame = Image.new("RGB", (224 * len(columns), 224 + header), "#101b20")
            draw = ImageDraw.Draw(frame)
            for column, path in enumerate(paths):
                with Image.open(path) as source:
                    if source.size != (224, 224):
                        raise ValueError(f"Unexpected model frame size: {source.size}")
                    frame.paste(source.convert("RGB"), (224 * column, header))
                if labels:
                    draw.text((224 * column + 8, 5), labels[column], font=font, fill="#f0f3ed")
            frame.save(temporary / f"{index:03d}.png")
            if poster and index == 0:
                frame.save(target.with_suffix(".webp"), quality=95)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "4",
            "-i", str(temporary / "%03d.png"), "-frames:v", "16", "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
            "-movflags", "+faststart", str(target),
        ], check=True)


def finetuned_samples(manifest, protocol):
    result = []
    for number in (7, 8, 9):
        matches = [sample for sample in manifest["samples"]
                   if sample.get("dataset") == "LuSNAR" and sample.get("scene") == f"Moon_{number}"
                   and sample.get("split") == "cross_scene_test" and sample.get("window") == 0
                   and sample.get("protocol") == protocol]
        if len(matches) != 1 or matches[0]["prediction_steps"] != list(range(1, 17)):
            raise ValueError(f"Expected one complete sixteen-step {protocol} window for Moon_{number}")
        result.append(matches[0])
    return result


def completed_eval(path):
    manifest = read_json(path)
    if not manifest.get("completed_utc"):
        raise ValueError(f"Inference has not completed: {path}")
    return manifest


def matching_protocol(left, right, same_checkpoint=False):
    keys = ["seed", "diffusion_steps", "vae_encoding", "image_size", "context_size"]
    if same_checkpoint:
        keys.append("checkpoint_sha256")
    for key in keys:
        if left["signature"][key] != right["signature"][key]:
            raise ValueError(f"Model comparison protocols disagree on {key}")


def matching_sample(left, right, ground_truth=True):
    keys = ["prediction_steps", "actions", "noise_key", "context_source_paths"]
    if ground_truth:
        keys += ["gt_source_paths", "gt_timestamps_ns", "waypoint_spacing_meters"]
    for key in keys:
        if left[key] != right[key]:
            raise ValueError(f"Model comparison samples disagree on {key}: {left['id']}")
    for key in (["context_paths", "gt_paths"] if ground_truth else ["context_paths"]):
        if [sha256(path) for path in left[key]] != [sha256(path) for path in right[key]]:
            raise ValueError(f"Model comparison pixels disagree on {key}: {left['id']}")


def public_model(manifest, name):
    training = read_json(Path(manifest["checkpoint"]).parent.parent / "metrics.json")
    if training.get("status") != "complete":
        raise ValueError("Only completed fine-tuned models can be compared")
    return {"name": name, "checkpoint_sha256": manifest["signature"]["checkpoint_sha256"],
            "source_checkpoint_sha256": training["source_checkpoint_sha256"],
            "training": {"settings": training["settings"], "windows": training["split_windows"]["train"],
                         "target_pairs": training["training_pairs"], "data_manifest_sha256": training["data_manifest_sha256"]}}


def publish_assets(directory, public_directory):
    public_directory.mkdir(parents=True, exist_ok=True)
    for asset in directory.iterdir():
        shutil.copyfile(asset, public_directory / asset.name)


def write_public_manifest(path, value):
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if "/file_system/" in serialized or '"/tmp/' in serialized:
        raise ValueError("Public provenance contains a machine-local path")
    write_json(path, value)


def export_finetuned(direct_path, autoregressive_path):
    direct, autoregressive = read_json(direct_path), read_json(autoregressive_path)
    for manifest in (direct, autoregressive):
        if not manifest.get("completed_utc"):
            raise ValueError("Only completed inference manifests can be published")
    signature = direct["signature"]
    for key in ("checkpoint_sha256", "seed", "diffusion_steps", "vae_encoding", "image_size", "context_size"):
        if signature[key] != autoregressive["signature"][key]:
            raise ValueError(f"Direct and autoregressive inference disagree on {key}")
    checkpoint = Path(direct["checkpoint"])
    training = read_json(checkpoint.parent.parent / "metrics.json")
    if training.get("status") != "complete":
        raise ValueError("Fine-tuning is not complete")
    if (training["settings"]["steps"], training["split_windows"]["train"], training["training_pairs"]) != (1000, 30, 480):
        raise ValueError("This publication protocol expects 1,000 updates on 30 windows / 480 target pairs")
    if (signature["seed"], signature["diffusion_steps"], signature["vae_encoding"],
            signature["context_size"], signature["image_size"]) != (0, 250, "posterior_mean", 4, 224):
        raise ValueError("Unexpected inference protocol for the fine-tuned website examples")
    data_root = Path(signature["lusnar_root"])

    def source_path(path):
        return "LuSNAR/" + Path(path).relative_to(data_root).as_posix()

    scenes = []
    # NAS holds temporary PNGs/encodes; only finished public media enters the site.
    with tempfile.TemporaryDirectory(prefix="website-export-", dir=direct_path.parent) as temporary:
        temporary = Path(temporary)
        for sample, ar_sample in zip(finetuned_samples(direct, "direct"),
                                     finetuned_samples(autoregressive, "autoregressive")):
            for key in ("context_source_paths", "gt_source_paths", "context_timestamps_ns",
                        "gt_timestamps_ns", "waypoint_spacing_meters"):
                if sample[key] != ar_sample[key]:
                    raise ValueError(f"Direct and autoregressive windows disagree on {key}")
            for key in ("context_paths", "gt_paths"):
                if [sha256(path) for path in sample[key]] != [sha256(path) for path in ar_sample[key]]:
                    raise ValueError(f"Direct and autoregressive images disagree on {key}")
            number = sample["scene"].split("_")[-1]
            target = temporary / f"moon-{number}"
            target.mkdir()
            Image.open(sample["initial_path"]).save(target / "initial.webp", quality=95)
            trajectory_svg([[0, 0]] + [[-left, forward] for forward, left, _ in sample["metric_actions"]],
                           target / "trajectory.svg")
            render_video([sample["prediction_paths"]], target / "direct.mp4", temporary)
            render_video([sample["gt_paths"]], target / "ground-truth.mp4", temporary)
            render_video([sample["gt_paths"], sample["prediction_paths"]], target / "comparison.mp4",
                         temporary, ("RECORDED REFERENCE", "OPENNWM / DIRECT"))
            render_video([ar_sample["gt_paths"], ar_sample["prediction_paths"]], target / "autoregressive.mp4",
                         temporary, ("RECORDED REFERENCE", "OPENNWM / AUTOREGRESSIVE"))
            provenance = {
                "schema_version": 1, "model": "OpenNWM Nav15 Stage 2, LuSNAR fine-tuned EMA",
                "checkpoint_sha256": signature["checkpoint_sha256"],
                "source_checkpoint_sha256": training["source_checkpoint_sha256"],
                "training": {"settings": training["settings"], "windows": training["split_windows"]["train"],
                             "target_pairs": training["training_pairs"], "scenes": [f"Moon_{i}" for i in range(1, 6)],
                             "data_manifest_sha256": training["data_manifest_sha256"]},
                "sample_id": sample["id"], "dataset": "LuSNAR", "scene": sample["scene"],
                "split": "cross_scene_test", "window": 0,
                "selection": "First preselected held-out window in each of Moon_7, Moon_8 and Moon_9; not ranked by prediction quality.",
                "context_source_paths": [source_path(path) for path in sample["context_source_paths"]],
                "gt_source_paths": [source_path(path) for path in sample["gt_source_paths"]],
                "pose_source": source_path(sample["pose_provenance"]),
                "source_sha256": {source_path(path): value for path, value in sample["source_sha256"].items()},
                "context_timestamps_ns": sample["context_timestamps_ns"],
                "gt_timestamps_ns": sample["gt_timestamps_ns"],
                "actual_prediction_offsets_seconds": sample["actual_prediction_offsets_seconds"],
                "prediction_steps": sample["prediction_steps"], "playback_fps": 4, "frame_interpolation": False,
                "timing": "Nearest-source sampling on a nominal 4 Hz grid; actual offsets above retain nonuniform capture times.",
                "sampling": {key: signature[key] for key in ("seed", "diffusion_steps", "vae_encoding", "image_size", "context_size")},
                "direct": {"protocol": "Each horizon is predicted from the same four observed context frames and anchor-relative target body pose.",
                           "actions": sample["actions"], "metric_actions": sample["metric_actions"],
                           "inference_manifest_sha256": sha256(direct_path)},
                "autoregressive": {"protocol": "Generated RGB frames are re-encoded and fed back; actions refer to the immediately preceding body pose.",
                                    "actions": ar_sample["actions"], "metric_actions": ar_sample["metric_actions"],
                                    "inference_manifest_sha256": sha256(autoregressive_path)},
                "waypoint_spacing_meters": sample["waypoint_spacing_meters"],
                "action_translation_units": sample["action_translation_units"],
                "action_yaw_units": sample["action_yaw_units"],
                "coordinate_convention": sample["coordinate_convention"],
                "limitations": "Qualitative held-out simulated scenes; direct predictions are not a continuous autoregressive rollout. Geometry and long-rollout consistency remain imperfect. Not a validation of physical navigation.",
            }
            if "/file_system/" in json.dumps(provenance):
                raise ValueError("Public provenance contains a machine-local path")
            public_directory = OUT / "lusnar-finetuned" / target.name
            previous_manifest = public_directory / "manifest.json"
            if previous_manifest.exists():
                comparison = read_json(previous_manifest).get("model_comparison")
                if comparison:
                    if comparison["opennwm"]["checkpoint_sha256"] != signature["checkpoint_sha256"]:
                        raise ValueError("Existing model comparison uses a different OpenNWM checkpoint")
                    if (comparison["direct_manifest_sha256"] != sha256(direct_path)
                            or comparison["autoregressive_manifest_sha256"] != sha256(autoregressive_path)):
                        raise ValueError("Existing model comparison uses different OpenNWM predictions")
                    provenance["model_comparison"] = comparison
            write_json(target / "manifest.json", provenance)
            public_directory.mkdir(parents=True, exist_ok=True)
            for asset in target.iterdir():
                shutil.copyfile(asset, public_directory / asset.name)
            prefix = public_directory.relative_to(SITE).as_posix()
            scenes.append({
                "id": FINETUNED_PREFIX + f"moon-{number}", "title": f"Moon {number} · fine-tuned",
                "titleZh": f"Moon {number} · fine-tuned", "tag": "LUSNAR · FINETUNED", "tagZh": "LUSNAR · FINETUNED",
                "description": "An unseen simulated lunar scene after small-sample adaptation. Each future view is predicted from four observed frames and the recorded target pose.",
                "descriptionZh": "An unseen simulated lunar scene after small-sample adaptation. Each future view is predicted from four observed frames and the recorded target pose.",
                "initial": f"{prefix}/initial.webp", "poster": f"{prefix}/initial.webp", "video": f"{prefix}/direct.mp4",
                "groundTruth": f"{prefix}/ground-truth.mp4", "comparison": f"{prefix}/comparison.mp4",
                "autoregressive": f"{prefix}/autoregressive.mp4", "trajectory": f"{prefix}/trajectory.svg",
                "manifest": f"{prefix}/manifest.json", "predictionLabel": "FINETUNED · DIRECT PREDICTION",
                "predictionLabelZh": "FINETUNED · DIRECT PREDICTION", "comparisonLabel": "Recorded reference / direct prediction",
                "comparisonLabelZh": "Recorded reference / direct prediction", "autoregressiveLabel": "Recorded reference / autoregressive rollout",
                "autoregressiveLabelZh": "Recorded reference / autoregressive rollout",
                "provenance": "OpenNWM Nav15 Stage 2, fine-tuned for 1,000 steps on 30 LuSNAR windows (480 target pairs). Moon 7–9 were held out. Four observed context frames; direct horizons 1–16; DDPM 250, seed 0, VAE posterior means. All 16 predictions play at 4 FPS without interpolation. Recorded capture offsets are retained in the manifest. Lunar appearance improves, but geometry and autoregressive consistency remain imperfect; this is not a physical-navigation result.",
                "provenanceZh": "OpenNWM Nav15 Stage 2, fine-tuned for 1,000 steps on 30 LuSNAR windows (480 target pairs). Moon 7–9 were held out. Four observed context frames; direct horizons 1–16; DDPM 250, seed 0, VAE posterior means. All 16 predictions play at 4 FPS without interpolation. Recorded capture offsets are retained in the manifest. Lunar appearance improves, but geometry and autoregressive consistency remain imperfect; this is not a physical-navigation result.",
            })
    catalog_path = SITE / "content/space.json"
    catalog = read_json(catalog_path)
    previous = {scene["id"]: scene for scene in catalog["scenes"]}
    catalog["scenes"] = [{**previous.get(scene["id"], {}), **scene} for scene in scenes] + [scene for scene in catalog["scenes"]
                                  if not scene["id"].startswith(FINETUNED_PREFIX)]
    write_json(catalog_path, catalog)
    print(f"Published {len(scenes)} fine-tuned LuSNAR examples; preserved all other scenes.")


def export_model_comparisons(direct_path, autoregressive_path, nwm_path, nwm_ar_path, single_path):
    direct, autoregressive = completed_eval(direct_path), completed_eval(autoregressive_path)
    nwm, nwm_ar, single = completed_eval(nwm_path), completed_eval(nwm_ar_path), completed_eval(single_path)
    matching_protocol(direct, nwm)
    matching_protocol(direct, single, same_checkpoint=True)
    matching_protocol(nwm, nwm_ar, same_checkpoint=True)
    models = {"opennwm": public_model(direct, "OpenNWM Nav15 Stage 2, LuSNAR fine-tuned EMA"),
              "nwm_release": public_model(nwm, "NWM-release, LuSNAR fine-tuned EMA")}
    if models["opennwm"]["training"] != models["nwm_release"]["training"]:
        raise ValueError("Model comparison requires identical training data and budgets")
    catalog_path = SITE / "content/space.json"
    catalog = read_json(catalog_path)
    scenes = {scene["id"]: scene for scene in catalog["scenes"]}
    with tempfile.TemporaryDirectory(prefix="website-comparison-", dir=nwm_path.parent) as temporary:
        temporary = Path(temporary)
        for sample, ar_sample, nwm_sample, nwm_ar_sample in zip(
                finetuned_samples(direct, "direct"), finetuned_samples(autoregressive, "autoregressive"),
                finetuned_samples(nwm, "direct"), finetuned_samples(nwm_ar, "autoregressive")):
            matching_sample(sample, nwm_sample)
            matching_sample(ar_sample, nwm_ar_sample)
            number = sample["scene"].split("_")[-1]
            scene = scenes[FINETUNED_PREFIX + f"moon-{number}"]
            target = temporary / f"moon-{number}"
            target.mkdir()
            render_video([sample["gt_paths"], sample["prediction_paths"], nwm_sample["prediction_paths"]],
                         target / "model-comparison.mp4", temporary,
                         ("RECORDED REFERENCE", "OPENNWM / FINETUNED", "NWM / FINETUNED"), poster=True)
            render_video([ar_sample["gt_paths"], ar_sample["prediction_paths"], nwm_ar_sample["prediction_paths"]],
                         target / "ar-model-comparison.mp4", temporary,
                         ("RECORDED REFERENCE", "OPENNWM / FINETUNED", "NWM / FINETUNED"))
            provenance = read_json(SITE / scene["manifest"])
            provenance["model_comparison"] = {
                **models, "direct_manifest_sha256": sha256(direct_path),
                "nwm_direct_manifest_sha256": sha256(nwm_path),
                "autoregressive_manifest_sha256": sha256(autoregressive_path),
                "nwm_autoregressive_manifest_sha256": sha256(nwm_ar_path),
                "conditions": "Identical observed frames, raw body actions, horizons, seed and diffusion noise keys. Each model uses its native action interface.",
            }
            write_public_manifest(target / "manifest.json", provenance)
            public_directory = OUT / "lusnar-finetuned" / target.name
            publish_assets(target, public_directory)
            prefix = public_directory.relative_to(SITE).as_posix()
            scene.update({"fineTuned": True, "modelComparison": f"{prefix}/model-comparison.mp4",
                          "modelComparisonPoster": f"{prefix}/model-comparison.webp",
                          "modelComparisonLabel": "Reference / OpenNWM / NWM · fine-tuned direct predictions",
                          "modelComparisonLabelZh": "Reference / OpenNWM / NWM · fine-tuned direct predictions",
                          "arModelComparison": f"{prefix}/ar-model-comparison.mp4",
                          "arModelComparisonLabel": "Reference / OpenNWM / NWM · fine-tuned autoregressive rollouts",
                          "arModelComparisonLabelZh": "Reference / OpenNWM / NWM · fine-tuned autoregressive rollouts"})
        export_single_image_comparisons(single, nwm, single_path, nwm_path, models, scenes, temporary)
    write_json(catalog_path, catalog)
    print("Published fine-tuned OpenNWM / NWM comparisons for all five lunar examples.")


def export_single_image_comparisons(single, nwm, single_path, nwm_path, models, scenes, temporary):
    for scene_id in ("moon-yutu2", "moon-lusnar"):
        pairs = [[sample for sample in manifest["samples"] if sample.get("source_scene_id") == scene_id]
                 for manifest in (single, nwm)]
        if any(len(matches) != 1 for matches in pairs):
            raise ValueError(f"Expected exactly one single-image example per model: {scene_id}")
        sample, nwm_sample = (matches[0] for matches in pairs)
        matching_sample(sample, nwm_sample, ground_truth=False)
        if sample["protocol"] != "autoregressive" or sample["prediction_steps"] != list(range(1, 17)):
            raise ValueError("Single-image examples require all sixteen autoregressive steps")
        if sample["gt_source_paths"] or len(set(sample["context_source_paths"])) != 1:
            raise ValueError("Single-image synthetic paths must have one initial source and no paired truth")
        legacy = read_json(SITE / f"assets/playground/{scene_id}/forward.json")
        if sample["source_details"] != legacy["source"] or sample["actions"] != legacy["actions"]:
            raise ValueError("Updated single-image examples must preserve the original source and controls")
        if sample["source_sha256"][sample["context_source_paths"][0]] != legacy["source"]["sha256"]:
            raise ValueError("Single-image source hash disagrees with its published provenance")
        target = temporary / scene_id
        target.mkdir()
        Image.open(sample["initial_path"]).save(target / "initial.webp", quality=95)
        trajectory_svg(sample["points"], target / "trajectory.svg")
        render_video([sample["prediction_paths"]], target / "opennwm.mp4", temporary)
        render_video([sample["prediction_paths"], nwm_sample["prediction_paths"]],
                     target / "model-comparison.mp4", temporary,
                     ("OPENNWM / FINETUNED", "NWM / FINETUNED"), poster=True)
        source = sample["source_details"]
        collection = "LuSNAR" if sample["dataset"] == "LuSNAR" else "planetary_rover"
        relative_source = f"{collection}/{source['trajectory']}/{Path(sample['context_source_paths'][0]).name}"
        provenance = {
            "schema_version": 1, "sample_id": sample["id"], "source_scene_id": scene_id,
            "dataset": sample["dataset"], "source": source, "context_source_paths": [relative_source] * 4,
            "source_sha256": {relative_source: source["sha256"]}, "split": sample["split"],
            "scene_seen_in_training": sample["scene_seen_in_training"],
            "source_used_in_fine_tuning": bool(sample["fine_tuning_source_overlap"]),
            "protocol": "Single source image repeated into four context frames; autoregressive RGB feedback with VAE posterior means; synthetic straight-path controls.",
            "prediction_steps": sample["prediction_steps"], "playback_fps": 4, "frame_interpolation": False,
            "sampling": {key: single["signature"][key] for key in ("seed", "diffusion_steps", "vae_encoding", "image_size", "context_size")},
            "noise_key": sample["noise_key"], "points": sample["points"], "actions": sample["actions"],
            "action_reference": sample["action_reference"], "action_translation_units": sample["action_translation_units"],
            "action_yaw_units": sample["action_yaw_units"], "coordinate_convention": sample["coordinate_convention"],
            "waypoint_spacing_meters": None, "timing": sample["timing"], "paired_future_ground_truth": False,
            "model_comparison": {**models, "opennwm_manifest_sha256": sha256(single_path),
                                 "nwm_manifest_sha256": sha256(nwm_path),
                                 "conditions": "Identical initial image, raw controls, horizons, seed and diffusion noise keys; native model action interfaces. The noise and RGB-feedback protocol differs from archived pre-fine-tuning website demos; no pixel-identical reproduction of those older demos is claimed."},
            "limitations": sample["limitations"],
        }
        write_public_manifest(target / "manifest.json", provenance)
        public_directory = OUT / "lunar-finetuned" / scene_id
        publish_assets(target, public_directory)
        prefix = public_directory.relative_to(SITE).as_posix()
        simulated = scene_id == "moon-lusnar"
        domain_note = ("Moon_1 was a fine-tuning scene, so this is a qualitative example rather than a held-out benchmark."
                       if simulated else "The real Yutu-2 image tests transfer from fine-tuning on simulated LuSNAR only.")
        scene = scenes[scene_id]
        scene.update({
            "fineTuned": True, "tag": "LUSNAR · FINETUNED" if simulated else "YUTU-2 · FINETUNED",
            "tagZh": "LUSNAR · FINETUNED" if simulated else "YUTU-2 · FINETUNED",
            "description": f"Fine-tuned OpenNWM and NWM share the same initial image and forward controls. {domain_note}",
            "descriptionZh": f"Fine-tuned OpenNWM and NWM share the same initial image and forward controls. {domain_note}",
            "initial": f"{prefix}/initial.webp", "poster": f"{prefix}/initial.webp", "video": f"{prefix}/opennwm.mp4",
            "modelComparison": f"{prefix}/model-comparison.mp4", "modelComparisonPoster": f"{prefix}/model-comparison.webp",
            "trajectory": f"{prefix}/trajectory.svg", "manifest": f"{prefix}/manifest.json",
            "predictionLabel": "FINETUNED · AUTOREGRESSIVE", "predictionLabelZh": "FINETUNED · AUTOREGRESSIVE",
            "modelComparisonLabel": "OpenNWM / NWM · fine-tuned autoregressive rollouts",
            "modelComparisonLabelZh": "OpenNWM / NWM · fine-tuned autoregressive rollouts",
            "provenance": f"Both models were fine-tuned for 1,000 steps on the same 30 LuSNAR windows (480 target pairs). {domain_note} A single image is repeated into four context frames; synthetic forward action 0.75 per step, 16 autoregressive steps, DDPM 250, seed 0, VAE posterior means. Videos play at 4 FPS without interpolation. There is no paired future reference or calibrated distance/time. Appearance and geometry may drift; this is not a physical-navigation result.",
            "provenanceZh": f"Both models were fine-tuned for 1,000 steps on the same 30 LuSNAR windows (480 target pairs). {domain_note} A single image is repeated into four context frames; synthetic forward action 0.75 per step, 16 autoregressive steps, DDPM 250, seed 0, VAE posterior means. Videos play at 4 FPS without interpolation. There is no paired future reference or calibrated distance/time. Appearance and geometry may drift; this is not a physical-navigation result.",
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned-eval", type=Path, help="Completed dense direct inference manifest")
    parser.add_argument("--autoregressive-eval", type=Path, help="Completed autoregressive inference manifest")
    parser.add_argument("--nwm-eval", type=Path, help="Matching fine-tuned NWM dense and single-image manifest")
    parser.add_argument("--nwm-autoregressive-eval", type=Path, help="Matching fine-tuned NWM held-out autoregressive manifest")
    parser.add_argument("--single-image-eval", type=Path, help="Fine-tuned OpenNWM Yutu-2 and LuSNAR single-image manifest")
    args = parser.parse_args()
    if bool(args.finetuned_eval) != bool(args.autoregressive_eval):
        parser.error("--finetuned-eval and --autoregressive-eval must be supplied together")
    comparison_args = (args.nwm_eval, args.nwm_autoregressive_eval, args.single_image_eval)
    if any(comparison_args) and not (all(comparison_args) and args.finetuned_eval):
        parser.error("Model comparisons require both original evaluation options and all three comparison options")
    if args.finetuned_eval:
        export_finetuned(args.finetuned_eval, args.autoregressive_eval)
        if args.nwm_eval:
            export_model_comparisons(args.finetuned_eval, args.autoregressive_eval,
                                     args.nwm_eval, args.nwm_autoregressive_eval, args.single_image_eval)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    sample = ARCHIVE / "examples/planetary_rover/id_4"
    trajectory = json.loads((sample / "trajectory.json").read_text())
    # Keep data-relative provenance; the public export does not need host paths.
    trajectory["source_pose_file"] = f"planetary_rover/{trajectory['trajectory']}/traj_data.pkl"
    # This archived sample exposes the observed failure, with no frame selection
    # or interpolation. It remains available alongside the new drawn-path tests.
    for name, directory in (("mars-archive-prediction", "opennwm-finalLAM-100k"), ("mars-archive-gt", "GT")):
        target = OUT / f"{name}.mp4"
        if not target.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "4", "-i", str(sample / "frames" / directory / "%03d.png"), "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True)
    Image.open(ARCHIVE / "initial/planetary_rover/id_4.png").save(OUT / "mars-archive-initial.webp", quality=95)
    trajectory_svg([[-left, forward] for forward, left in trajectory["positions_local"]], OUT / "mars-archive-trajectory.svg")
    (OUT / "mars-archive-source.json").write_text(json.dumps({"archive": ARCHIVE.name, "sampleId": 4, "trajectory": trajectory, "protocol": "Single real image repeated to four context frames; autoregressive RGB re-encoding; DDPM 250, seed 0; OpenNWM Nav15 Stage2 joint100k.", "assessment": "Severe appearance drift into indoor scenes; this is a failure example, not evidence of successful planetary navigation.", "timing": "16 successive source observations displayed at 4 FPS, not regular real mission capture intervals."}, indent=2) + "\n")

    catalog_path = SITE / "content/space.json"
    previous = {scene["id"]: scene for scene in read_json(catalog_path)["scenes"]} if catalog_path.exists() else {}
    scenes = [scene for scene in previous.values()
              if scene["id"].startswith(FINETUNED_PREFIX)] if catalog_path.exists() else []
    definitions = [
        ("moon-yutu2", "The Moon", "The Moon", "YUTU-2 · REAL MISSION", "YUTU-2 · REAL MISSION", "A real lunar observation from Chang’e-4 / Yutu-2, conditioned on a forward path.", "A real lunar observation from Chang’e-4 / Yutu-2, conditioned on a forward path."),
        ("mars-zhurong", "Mars", "Mars", "ZHURONG · REAL MISSION", "ZHURONG · REAL MISSION", "A real Martian observation from Tianwen-1 / Zhurong, conditioned on a forward path.", "A real Martian observation from Tianwen-1 / Zhurong, conditioned on a forward path."),
        ("moon-lusnar", "A simulated Moon", "A simulated Moon", "LUSNAR · SIMULATION", "LUSNAR · SIMULATION", "A simulated lunar landscape from LuSNAR, evaluated with the same checkpoint.", "A simulated lunar landscape from LuSNAR, evaluated with the same checkpoint."),
    ]
    for scene_id, title, title_zh, tag, tag_zh, description, description_zh in definitions:
        if previous.get(scene_id, {}).get("fineTuned"):
            scenes.append(previous[scene_id])
            continue
        manifest_file = SITE / f"assets/playground/{scene_id}/forward.json"
        if not manifest_file.is_file():
            continue
        manifest = json.loads(manifest_file.read_text())
        prefix = f"assets/playground/{scene_id}"
        trajectory_svg(manifest["points"], OUT / f"{scene_id}-trajectory.svg")
        direct = manifest.get("mode") == "direct"
        protocol = "Each future frame is predicted directly from the fixed initial observation and its target pose." if direct else "Autoregressive prediction feeds each generated image back into the model. Severe appearance drift was observed in these planetary rollouts."
        timing = "16 model steps are displayed at 4 FPS; this simulation uses normalized planar controls, without metric calibration." if scene_id == "moon-lusnar" else "16 model steps are displayed at 4 FPS; real mission capture intervals are irregular."
        scenes.append(dict(id=scene_id, title=title, titleZh=title_zh, tag=tag, tagZh=tag_zh,
            description=description, descriptionZh=description_zh, initial=f"{prefix}/initial.webp", poster=f"{prefix}/initial.webp", video=f"{prefix}/forward.mp4", trajectory=f"assets/space/{scene_id}-trajectory.svg", manifest=f"{prefix}/forward.json",
            provenance=f"OpenNWM · Nav15 Stage 2 joint100k · DDPM 250 · seed 0 · VAE posterior mean. {protocol} These are qualitative stress tests, not proof of arbitrary-scene or physical navigation capability. {timing}",
            provenanceZh=f"OpenNWM · Nav15 Stage 2 joint100k · DDPM 250 · seed 0 · VAE posterior mean. {protocol} These are qualitative stress tests, not proof of arbitrary-scene or physical navigation capability. {timing}"))
    scenes.append(dict(id="mars-recorded", title="Recorded-path stress test", titleZh="Recorded-path stress test", tag="MARS · AUTOREGRESSIVE", tagZh="MARS · AUTOREGRESSIVE", description="Conditioned on the recorded rover path, appearance drifts away from the Martian scene. Open the recorded reference to inspect the failure.", descriptionZh="Conditioned on the recorded rover path, appearance drifts away from the Martian scene. Open the recorded reference to inspect the failure.", initial="assets/space/mars-archive-initial.webp", poster="assets/space/mars-archive-initial.webp", video="assets/space/mars-archive-prediction.mp4", groundTruth="assets/space/mars-archive-gt.mp4", trajectory="assets/space/mars-archive-trajectory.svg", provenance="Archived inference · 24 September 2026 · sample 4 · joint100k · DDPM 250 · seed 0. All 16 predictions are shown, without interpolation. Source intervals are irregular; playback is not real mission time.", provenanceZh="Archived inference · 24 September 2026 · sample 4 · joint100k · DDPM 250 · seed 0. All 16 predictions are shown, without interpolation. Source intervals are irregular; playback is not real mission time.", manifest="assets/space/mars-archive-source.json"))
    (SITE / "content/space.json").write_text(json.dumps({"schemaVersion": 1, "scenes": scenes}, ensure_ascii=False, indent=2) + "\n")
    print(f"Published {len(scenes)} traceable planetary examples.")


if __name__ == "__main__":
    main()

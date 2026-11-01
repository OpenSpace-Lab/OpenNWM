"""Generate genuine trajectory-conditioned presets and their public catalog."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from inference import Predictor, public_manifest
from scenes import SCENES, PLAYGROUND_SCENE_IDS, DEFAULT_SCENE, WEBSITE, DEFAULT_RUN, DEFAULT_OUTPUT, source_path
from trajectory import PRESETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--mode", choices=["auto", "autoregressive", "direct"], default="auto")
    parser.add_argument("--no-publish", action="store_true", help="Keep a diagnostic run on NAS without changing the website.")
    args = parser.parse_args()
    if args.scenes and set(args.scenes) - {scene["id"] for scene in SCENES}:
        parser.error("Unknown scene ID.")
    predictor = Predictor(args.run_dir, args.steps, None if args.mode == "auto" else args.mode)
    catalog = {"schemaVersion": 1, "model": "OpenNWM · Nav15 Stage 2 joint100k", "defaultScene": DEFAULT_SCENE, "fps": 4, "duration": 4,
        "coordinateSystem": {"x": "right", "y": "forward", "origin": [0, 0], "bounds": {"x": [-1, 1], "y": [-.25, 1.5]}, "canvasUnitsToWaypointUnits": 12},
        "note": "Presets are genuine OpenNWM predictions conditioned on the displayed path. Draw a new path with the local GPU service to generate a new prediction.",
        "noteZh": "Presets are genuine OpenNWM predictions conditioned on the displayed path. Draw a new path with the local GPU service to generate a new prediction.",
        "planetaryNote": "Moon and Mars start from real mission images. The 4-second horizon describes 16 model steps at 4 FPS; mission capture intervals are irregular.",
        "planetaryNoteZh": "Moon and Mars start from real mission images. The 4-second horizon describes 16 model steps at 4 FPS; mission capture intervals are irregular.", "scenes": []}
    labels = {"forward": ("Forward", "Forward"), "left": ("Turn left", "Turn left"), "right": ("Turn right", "Turn right")}
    for scene in SCENES:
        if args.scenes and scene["id"] not in args.scenes:
            continue
        if not args.scenes and scene["id"] not in PLAYGROUND_SCENE_IDS:
            continue
        directories = [Path(args.output) / "presets" / scene["id"] / key for key in PRESETS]
        requests = [{"points": points, "seed": 0} for points in PRESETS.values()]
        prediction_mode = scene.get("predictionMode", "autoregressive") if args.mode == "auto" else args.mode
        source_hash = hashlib.sha256(source_path(scene).read_bytes()).hexdigest()
        checkpoint = (Path(args.run_dir) / "checkpoints/latest.pth.tar").resolve()
        complete = True
        for request, directory in zip(requests, directories):
            try:
                saved = json.loads((directory / "manifest.json").read_text())
                complete &= ((directory / "prediction.mp4").is_file()
                    and saved.get("mode", "autoregressive") == prediction_mode
                    and saved["diffusionSteps"] == args.steps and saved["points"] == request["points"]
                    and saved["seed"] == request["seed"] and saved["source"]["sha256"] == source_hash
                    and saved["checkpoint"]["path"] == str(checkpoint)
                    and saved["checkpoint"]["mtimeNs"] == checkpoint.stat().st_mtime_ns)
            except (OSError, KeyError, ValueError):
                complete = False
        if not complete:
            predictor.predict(scene, requests, directories)
        if args.no_publish:
            print(f"Diagnostic outputs retained for {scene['id']} ({prediction_mode})", flush=True)
            continue
        public = {key: scene[key] for key in ("id", "title", "titleZh", "dataset", "domain", "description", "descriptionZh")}
        public["planetary"] = scene.get("planetary", False)
        public["simulated"] = scene.get("simulated", False)
        public["qualityNote"] = scene.get("qualityNote", "")
        public["qualityNoteZh"] = scene.get("qualityNoteZh", "")
        public["predictionMode"] = prediction_mode
        public["protocol"] = ("Fixed initial observation; direct prediction at each future path endpoint." if prediction_mode == "direct" else "Autoregressive prediction; each generated view becomes the next observation.")
        public["protocolZh"] = ("Fixed initial observation; direct prediction at each future path endpoint." if prediction_mode == "direct" else "Autoregressive prediction; each generated view becomes the next observation.")
        public["presets"] = []
        destination = WEBSITE / "assets/playground" / scene["id"]
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directories[0] / "initial.webp", destination / "initial.webp")
        public["poster"] = public["image"] = f"assets/playground/{scene['id']}/initial.webp"
        for (key, points), directory in zip(PRESETS.items(), directories):
            for source_name, name in (("prediction.mp4", f"{key}.mp4"), ("poster.webp", f"{key}.webp")):
                shutil.copyfile(directory / source_name, destination / name)
            manifest = public_manifest(json.loads((directory / "manifest.json").read_text()))
            # Initial autoregressive archives predate the explicit mode field.
            manifest.setdefault("mode", "autoregressive")
            manifest.setdefault("modelConditions", manifest["actions"])
            manifest.setdefault("simulatedSource", scene.get("simulated", False))
            (destination / f"{key}.json").write_text(json.dumps(manifest, indent=2) + "\n")
            public["presets"].append({"id": key, "label": labels[key][0], "labelZh": labels[key][1], "points": points,
                "video": f"assets/playground/{scene['id']}/{key}.mp4", "poster": f"assets/playground/{scene['id']}/{key}.webp",
                "manifest": f"assets/playground/{scene['id']}/{key}.json", "duration": 4, "seed": 0, "diffusionSteps": manifest["diffusionSteps"]})
        catalog["scenes"].append(public)
        catalog_path = WEBSITE / "content/playground.json"
        if args.scenes and catalog_path.is_file():
            previous = json.loads(catalog_path.read_text())
            updated = {entry["id"]: entry for entry in previous["scenes"]}
            updated.update({entry["id"]: entry for entry in catalog["scenes"]})
            catalog["scenes"] = [updated[scene_id] for scene_id in PLAYGROUND_SCENE_IDS if scene_id in updated]
        else:
            updated = {entry["id"]: entry for entry in catalog["scenes"]}
            catalog["scenes"] = [updated[scene_id] for scene_id in PLAYGROUND_SCENE_IDS if scene_id in updated]
        temporary = catalog_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(catalog_path)
        print(f"Published {scene['id']}: three real presets", flush=True)


if __name__ == "__main__":
    main()

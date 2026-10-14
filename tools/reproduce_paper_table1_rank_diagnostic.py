#!/usr/bin/env python3
"""Compare the first historical batch under unchanged and seed-zero VAE RNG.

Diagnostic only: preserve all dataset construction and the original first batch,
write separate variants, then exit normally. No regular replay code is changed.
Use the same arguments as reproduce_paper_table1.py infer.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import reproduce_paper_table1 as replay


class DiagnosticComplete(BaseException):
    pass


def entry(args):
    metadata = json.loads(args.metadata.read_text())
    if metadata.get("diagnostic_local_rank") is not None:
        os.environ["LOCAL_RANK"] = str(metadata["diagnostic_local_rank"])
    runtime = Path(metadata["runtime"])
    sys.path.insert(0, str(runtime))
    import isolated_nwm_infer as inference
    import torch

    run_dir = Path(metadata["run_directory"])
    records = {"created_utc": replay.now(), "events": [], "datasets_created": []}

    def digest(tensor):
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def snapshot(event, **extra):
        records["events"].append({
            "event": event,
            "cuda_device": torch.cuda.current_device(),
            "cuda_initial_seed": torch.cuda.initial_seed(),
            "cuda_rng_sha256": digest(torch.cuda.get_rng_state()),
            "cuda_rng_prefix": torch.cuda.get_rng_state()[:24].tolist(),
            "cpu_rng_sha256": digest(torch.get_rng_state()),
            **extra,
        })
        replay.write_json(run_dir / "diagnostic.json", records)

    original_seed = inference.misc.seed_everything

    def seed(value):
        original_seed(value)
        snapshot("after_seed_everything", requested_seed=int(value))

    inference.misc.seed_everything = seed
    original_tokenizer = inference.setup_tokenizer

    def tokenizer(*positional, **keyword):
        result = original_tokenizer(*positional, **keyword)
        snapshot("after_setup_tokenizer")
        original_encode = result.encode

        def encode(x):
            snapshot("before_vae_encode", input_sha256=digest(x), shape=list(x.shape))
            value = original_encode(x)
            snapshot("after_vae_encode", latent_sha256=digest(value))
            return value

        result.encode = encode
        return result

    inference.setup_tokenizer = tokenizer
    original_model = inference.setup_model

    def model(*positional, **keyword):
        result = original_model(*positional, **keyword)
        snapshot("after_setup_model")
        return result

    inference.setup_model = model
    original_dataset = inference.EvalDataset

    def dataset(*positional, **keyword):
        result = original_dataset(*positional, **keyword)
        records["datasets_created"].append(keyword["dataset_name"])
        return result

    inference.EvalDataset = dataset
    original_generate = inference.generate_time

    def generate(config, output_dir, idxs, all_models, obs_image, gt_output,
                 delta, secs, num_cond, device, dataset_name):
        indices = [int(value) for value in idxs.reshape(-1).tolist()]
        expected = list(range(metadata["virtual_rank"], 128, 4))
        if dataset_name != "recon" or indices != expected:
            raise RuntimeError(f"Unexpected first batch: {dataset_name}, {indices}")
        variants = ["unchanged"]
        if not metadata.get("diagnostic_skip_seed0_variant", False):
            variants.append("vae_seed0")
        records.update(sample_indices=indices, batch_size=len(indices), variants=variants)
        snapshot("before_first_batch", actions_sha256=digest(delta),
                 observations_sha256=digest(obs_image))
        original_generate(config, output_dir, idxs, all_models, obs_image,
                          gt_output, delta, secs, num_cond, device, dataset_name)
        snapshot("after_unchanged_variant")
        if "vae_seed0" in variants:
            torch.cuda.manual_seed(0)
            snapshot("after_diagnostic_cuda_seed0")
            variant_dir = run_dir / "variant_vae_seed0" / dataset_name / "time"
            variant_dir.mkdir(parents=True)
            original_generate(config, str(variant_dir), idxs, all_models, obs_image,
                              gt_output, delta, secs, num_cond, device, dataset_name)
            snapshot("after_vae_seed0_variant")
        records["completed_utc"] = replay.now()
        replay.write_json(run_dir / "diagnostic.json", records)
        raise DiagnosticComplete

    inference.generate_time = generate
    try:
        replay.entry(args)
    except DiagnosticComplete:
        torch.distributed.destroy_process_group()
        print("Diagnostic completed normally after the first 32 samples; "
              "all original dataset constructors ran.", flush=True)
        return 0
    raise RuntimeError("Diagnostic did not intercept the first batch")


def infer(args):
    if args.cuda_visible_devices is not None:
        if args.physical_local_rank is None:
            raise ValueError("An alternate visible mapping requires --physical-local-rank")
        visible = args.cuda_visible_devices.split(",")
        if visible[args.physical_local_rank] != args.gpu:
            raise ValueError("The selected local device must map to the preflight --gpu")
    # This diagnostic deliberately permits a mismatched CUDA index, with
    # diagnostic-only protocol metadata; normal replay refuses such a shard.
    metadata = replay.prepare(args, enforce_historical_cuda_index=False)
    run_dir = Path(metadata["run_directory"])
    metadata["protocol"] = "diagnostic_only_first_original_batch_two_vae_rng_variants"
    metadata["command"][5] = str(Path(__file__).resolve())
    metadata["shell_command"] = replay.shlex.join(metadata["command"])
    metadata["diagnostic_sha256"] = replay.sha256(__file__)
    metadata["diagnostic_skip_seed0_variant"] = args.skip_seed0_variant
    if args.cuda_visible_devices is not None:
        metadata["environment_overrides"]["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        metadata["diagnostic_local_rank"] = args.physical_local_rank
    replay.write_json(run_dir / "metadata.json", metadata)
    print(json.dumps({key: metadata[key] for key in
                      ("runtime", "shell_command", "log", "gpu_preflight")}, indent=2),
          flush=True)
    environment = dict(os.environ, **metadata["environment_overrides"])
    with Path(metadata["log"]).open("w") as log:
        process = subprocess.Popen(metadata["command"], cwd=metadata["runtime"],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        metadata.update(state="running", pid=process.pid, started_utc=replay.now())
        replay.write_json(run_dir / "metadata.json", metadata)
        print(f"PID={process.pid}; log={metadata['log']}", flush=True)
        code = process.wait()
    metadata.update(state="completed" if code == 0 else "failed", exit_code=code,
                    finished_utc=replay.now())
    replay.write_json(run_dir / "metadata.json", metadata)
    print(f"Exit status: {code}; log tail:", flush=True)
    print("\n".join(Path(metadata["log"]).read_text(errors="replace").splitlines()[-15:]))
    return code


if __name__ == "__main__":
    parser = replay.parser()
    infer_parser = parser._subparsers._group_actions[0].choices["infer"]
    infer_parser.add_argument("--skip-seed0-variant", action="store_true")
    args = parser.parse_args()
    if args.operation not in ("infer", "_entry"):
        raise ValueError("Only infer and the internal _entry are supported")
    raise SystemExit({"infer": infer, "_entry": entry}[args.operation](args))

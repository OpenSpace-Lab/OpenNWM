# Validation record

Run this after the initial private GitHub push. Record the commit, commands,
environment, exit status and report paths here or in the run directory.

| Check | Required evidence | Result |
| --- | --- | --- |
| LAM original/new parity | Four variants: initial state, training outputs/losses, every gradient, two optimizer updates, eval outputs | Not run |
| NWM original/new parity | Time/geometry/IDM/latent/latent-only pretraining; reset/alignment/real-to-latent/controller warmup and joint; scratch baseline; production inference wrapper and 12-step DDPM sampling | Not run |
| LAM train/eval/resume | Tests in `lam/tests`, including periodic evaluation and checkpoint continuation | Not run |
| NWM train/eval/resume | Tests in `nwm/tests`, including production diffusion steps, phase transitions, RNG and data cursor restoration | Not run |

From the new repository root, using the **same interpreter for each old/new pair**:

```bash
/path/to/lam/python tools/compare_original.py --project lam \
  --original /path/to/original/lam_project --output-dir /path/to/artifacts/parity-lam
conda run --no-capture-output -n nwm python tools/compare_original.py --project nwm \
  --original /path/to/original/CompACT --output-dir /path/to/artifacts/parity-nwm
```

The tool selects `lam/` or `nwm/` in this repository as the candidate, starts two
isolated CPU processes, and writes `original.pt`, `candidate.pt`, two logs and
`report.json`. The candidate loads the original initial weights strictly; its
independent initialization is compared too. Default tolerances are zero. A
nonzero exit status means a mismatch or a worker failure. The old checkouts are
needed only for this migration check, never for ordinary training or evaluation.

Run each test suite from its own project directory so its modules and config
roots resolve independently:

```bash
cd lam
/path/to/lam/python -m pytest --confcutdir=tests tests -q
```

```bash
cd nwm
conda run --no-capture-output -n nwm python -m pytest --confcutdir=tests tests -q
```

Install pytest in a dedicated test environment if it is absent. On this machine,
the original LAM interpreter is
`/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/.venv/bin/python`.
The original NWM conda environment is `nwm`; it initially lacked pytest.

These checks establish implementation parity on deterministic small inputs.
The DINO comparison uses precomputed features, and the NWM comparison uses an
identity tokenizer. Existing tests cover image/data transforms and evaluation
adapters separately. They do not establish full-dataset metric equality,
external pretrained teacher/VAE parity, distributed GPU behavior, or convergence
of a complete training run. Record any subsequent real-checkpoint and dataset
evaluation separately with the checkpoint, config, split, seed and result.

## Shared validation split

```bash
conda run -n nwm python tools/make_validation_split.py \
  --output /path/to/shared-splits --seed 32 --val-fraction 0.05
```

This reads only small trajectory lists and fixed test indices. It holds out 5%
of the original training trajectories for `recon`, `scand`, `sacson` and
`tartan_drive`, preserves existing test indices, and leaves evaluation-only
datasets such as `go_stanford` with their test split. It does not open images or
modify the original splits. Reuse the output for both LAM and NWM. This optional
holdout changes the data recipe compared with training on the complete original
train lists; compare runs only when they use the same split directory.

## Optional real CLI smoke fixture

`tools/make_smoke_fixture.py --output /path/to/fixture` writes 240 tiny RGB
frames across 12 independent train/validation/test trajectories, plus two small
LAM configs. It needs NumPy, Pillow and PyYAML. From the repository root, use the
LAM environment for these commands (replace `python` with its explicit path):

```bash
python tools/make_smoke_fixture.py --output /path/to/fixture
python lam/run.py train --variant 2 --profile legacy \
  --data-root /path/to/fixture/data --split-root /path/to/fixture/splits \
  --run-dir /path/to/fixture/run --config /path/to/fixture/lam_train.yaml \
  --accelerator cpu --precision 32-true --devices 1 --num-workers 0 \
  --max-steps 2 --warmup-steps 0 --eval-every 1 --save-every 1 \
  --periodic-every 2 --eval-samples 2 --eval-batch-size 2
```

Repeat the training command with `--max-steps 4 --resume` to exercise interruption
within the four-batch epoch. Check that the global step advances to four and the
evaluation/checkpoint files are present. Changing this smoke run's scheduler
budget does not establish bit-exact equivalence to an uninterrupted run.

```bash
python lam/run.py eval all --config /path/to/fixture/lam_eval.yaml \
  --checkpoint /path/to/fixture/run/checkpoints/latest.ckpt \
  --output-dir /path/to/fixture/evaluation --device cpu --precision 32 \
  --batch-size 2 --num-workers 0 --disable-lpips
```

Repeat the evaluation command to test reuse of completed batches. These commands
exercise the public CLIs and real Lightning loops; they do not run a large model,
download pretrained weights, or read an existing image dataset.

# Validation record

2026-09-26: tests began after the initial private GitHub push, commit `c7d75e8`.
The numerical comparisons and unit tests run on CPU. CLI tests also exercise a
real DINO teacher and NWM training/evaluation on GPU. Artifacts and full logs
are stored outside the source.
The parity artifact root is
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/20260926`.

| Check | Required evidence | Result |
| --- | --- | --- |
| LAM original/new parity | Four variants: initial state, training outputs/losses, every gradient, two optimizer updates, eval outputs | **Passed: 2,436 tensors, maximum difference 0** |
| CompACT/new NWM parity | Time/geometry/IDM/latent/latent-only pretraining; reset/alignment/real-to-latent/controller warmup and joint; scratch baseline; production inference wrapper and 12-step DDPM sampling | **Passed: 14 cases, 8,372 tensors, maximum difference 0** |
| Original Meta NWM/new parity | Actual original training block: two AdamW updates, model outputs, gradients, parameters, EMA; production 12-step DDPM wrapper with original noise/clipping; train/eval image and action transforms | **Passed: 587 tensors and 110 scalars, maximum difference 0** |
| LAM mid-epoch resume | Actual Lightning checkpoint restore; workers 0 and 2; consumed batch cursor, random training across an epoch boundary, model and Adam states | **4 passed** |
| LAM suite | All LAM tests plus the four additional mid-epoch resume tests, after final cleanup | **125 passed** |
| LAM public CLI | All four variants train, validate, save best/latest/periodic checkpoints, resume training and evaluate | **Passed; details below** |
| LAM CPU DDP | Two ranks, actual Pixel CLI, mid-epoch checkpoint restore across an epoch boundary, rank-zero validation/save each step | **Passed: steps 2 → 4, no hang** |
| Existing LAM checkpoint | Original ActionOnly step-100000 checkpoint on real RECON test images | **Passed: 9 pairs and visualization** |
| LAM → NWM cache | Original ActionOnly step-100000 checkpoint, real RECON images, NWM strict cache loading and repeat preparation | **Passed: 682 pairs, exact tensor equality and cache reuse** |
| SD-VAE → NWM cache | Real SANPO images, pretrained SD-VAE, dataset/loader/posterior sampling and repeated preparation | **Passed: 141 frames, cached tensors exactly preserved** |
| NWM train/eval suite | Production diffusion steps, phase transitions, RNG/data cursor restoration; evaluation/checkpoint management tests | **123 tests + 47 subtests passed: full 121-test suite, then 8 related tests including 2 new regression cases** |
| NWM public CLI | GPU Stage 1, Stage 2 and original training/resume; real pretrained inference, metrics and planning/resume | **Passed; details below** |
| Published NWM dependency pins | Independent Python environment, complete suite, numerical parity and production CLI | **All passed, including final GPU inference/metrics/planning and continuation** |
| Portable setup and downloads | Local paths and paths containing spaces; HTTP Range continuation and completed-file reuse | **Passed** |

Completed commands used the repository root
`/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM` as their working directory:

```bash
LAM_PY=/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/.venv/bin/python
NWM_PY=/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM/.venv-nwm/bin/python
ARTIFACTS=/file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/20260926

"$LAM_PY" tools/compare_original.py --project lam \
  --original /file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/external/lam_project \
  --output-dir "$ARTIFACTS/lam_parity"
"$NWM_PY" tools/compare_original.py --project nwm \
  --original /file_system/vepfs/algorithm/dujun.nie/code/CompACT \
  --output-dir "$ARTIFACTS/nwm_parity"
"$NWM_PY" tools/compare_original.py --project nwm_original \
  --original /file_system/vepfs/algorithm/dujun.nie/code/nwm \
  --output-dir "$ARTIFACTS/original_nwm_parity"
"$LAM_PY" -m pytest tests/test_lam_resume.py -q --confcutdir=tests \
  --basetemp "$ARTIFACTS/lam_resume_tmp"
```

These commands used `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. All exited **0**. Parent PIDs were respectively 1681726,
1681727, 1718929 and 1681728. Logs are `lam_parity.log`, `nwm_parity.log`,
`original_nwm_parity.log` and `lam_resume.log`; each parity directory contains
`report.json`. Reports include exact worker commands, imported source
hashes, environments and each case's comparison counts.

LAM used Python 3.10.20, PyTorch 2.6.0+cu124 and Lightning 2.6.5. NWM used Python
3.10.20 and PyTorch 2.15.0.dev20260817+cu126. Both parity runs used one CPU thread;
LAM took 12.75 seconds, CompACT NWM 23.03 seconds, original Meta NWM 18.87 seconds
and resume tests 13.20 seconds.

### Public CLI evidence

LAM artifacts are at `/file_system/nas/algorithm/dujun.nie/opennwm-tests/lam`.
All four variants completed two training steps, then `--resume` to step four.
Every `latest.ckpt` contains optimizer state and global step four. Each run kept
`best`, `latest`, and the last three periodic checkpoints (steps 2, 3 and 4).
The LAM model uses dimension 8 and one encoder/decoder block with independent
synthetic train/validation/test trajectories. DINO uses the real 303M ViT-L/16
teacher on GPU 2; the Pixel evaluation also tested real VGG-LPIPS weights.

| Evaluation | Coverage | Result/log |
| --- | --- | --- |
| Action | ID, 27 pairs, visualization | Exit 0; `eval1.log` |
| Pixel | ID, 27 pairs, real VGG-LPIPS | Exit 0; `eval2-lpips.log` |
| DINO | ID/OOD, 36 pairs; separate ID PCA visualization, 27 pairs | Exit 0; `eval3.log`, `eval3-visual.log` |
| Pixel+Action | ID/ID-CD/OOD/OOD-CD, 19 conditions, 171 pairs, visualization | Exit 0; `eval4.log` |
| Evaluation continuation | Delete the final completed batch for DINO and Pixel+Action, then rerun | Exit 0; reused 3/18 batches respectively, recomputed one, summaries exactly unchanged |

Commands are retained in `resume_all.sh` and `eval_resume.sh` beside the logs.
Training logs are `variant{1,2,3,4}.log`, resume logs `resume{1,2,3,4}.log`, and
continuation logs `eval_resume{3,4}.log`.
The additional CPU DDP test is retained in `ddp_cpu.sh` with `ddp2.log` and
`ddp4.log`; both two-rank train/resume commands exited 0.
The resumed checkpoint records global step four and both ranks' RNG states.
The final combined command, from the repository root,
`$LAM_PY -m pytest lam/tests tests/test_lam_resume.py -q`, completed with 125
passed in 12.54 seconds, exit 0 (`final-unit.log`). One obsolete asset-validation
test was removed during cleanup; `eval prepare-assets` was then exercised
successfully against the existing LPIPS/VGG cache.

The original ActionOnly checkpoint
`/file_system/nas/algorithm/dujun.nie/nwm/weights/navigation_lam/variant_1_action/action/checkpoints/step=100000.ckpt`
also completed formal ID evaluation on nine real RECON test pairs from trajectory
`jackal_2019-08-08-17-49-25_1_r00`, including visualization. Results are under
`real/action100k`; `action100k-real.log` records exit 0. This confirms existing
checkpoint/data compatibility; it is not a complete-dataset benchmark.

The same original ActionOnly checkpoint also generated 682 action pairs from a
108-frame real RECON trajectory, `jackal_2019-09-09-18-46-15_12_r01`, through
`nwm/prepare.py lam`. Its raw current-to-goal posterior means have shape
`[682, 32]`. `OfflineProxyStore(strict_loading=True)` read all 682 pairs exactly.
Repeating preparation left both the tensor file timestamp and full manifest
unchanged. `bridge/prepare.sh`, `bridge/result.json` and logs under the LAM
artifact root retain the command and checks.

The real SD-VAE encoder cached 141 frames from SANPO trajectory
`xFHjZynUy8kysoxYFZ26ie7xG_cjj8wU` in BF16. The mean and log-variance each have
shape `[141, 4, 28, 28]`. All four recipe samples loaded by `NavAnywhereDataset`
match the stored tensors exactly; batching and posterior sampling returned finite
latents with shape `[2, 8, 4, 28, 28]`. Repeating preparation preserved the tensor
timestamp and manifest. Evidence is in the LAM artifact root's `vae-bridge/`
directory; the original encoder log is `nwm-cli/cache-vae.log` under the parity
artifact root.

The NWM suite covers interrupted metric accumulation, complete training
checkpoints, retention, evaluation output identity and imported checkpoints.
The final suite and inference regression results are recorded below under
the published dependency versions.

NWM production CLI evidence is under the parity artifact root's `nwm-cli/`
directory. Job logs and reports retain exact commands, PIDs and exit statuses.
Earlier evaluation and online validation outputs preceded the clipping fix;
the fresh `final-*` runs below replace those outputs as final evidence.
Interrupted versus uninterrupted production training was compared in
`production-resume-parity.json`: Stage 1 is exactly equal for 279 tensors and
original NWM for 295 tensors. Stage 2 compares 297 tensors with maximum absolute
difference **5.996e-10**; all pass `rtol=0, atol=1e-8`, but Stage 2 is not bitwise
equal. This GPU resume check is separate from the zero-tolerance CPU
original/new implementation comparisons above. `train_smoke.py` and
`resume_reference.py` retain its commands.

### Published dependency versions

An independent package environment was built at
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/envs/nwm-stable`
with Python 3.10.20, PyTorch 2.6.0+cu124, TorchVision 0.21.0+cu124, NumPy 1.26.4,
SciPy 1.15.2 and OpenCV Headless 4.10.0.84. All **23 production pins** in
`nwm/requirements.txt` match. The final `uv pip check` checked 121 packages and
reported all compatible, exit 0 (`stable-pip-check-final.log`).
The complete NWM suite passed
**121 tests and 47 subtests in 84.40 seconds**, exit 0, including the final
DreamSim cache-location regression test. Command:

```bash
cd /file_system/vepfs/algorithm/dujun.nie/code/OpenNWM/nwm
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/envs/nwm-stable/bin/python -m pytest tests -q
```

Its PID was 1799202 and log is `stable-nwm-tests.log` under the parity artifact
root. `stable-environment.json` records installed versions and import locations.
Real LPIPS image loading exposed a missing OpenCV dependency in the initially
isolated environment. The published requirements now include
`opencv-python-headless==4.10.0.84`; the original failure is retained in
`nwm-cli/final-metric-missing-cv2.log`.

This was not a complete fresh download: slow network transfer was replaced by
independent copies of the already installed PyTorch, TorchVision, Triton and
matching NVIDIA packages from `DreamDojo/.venv`. No `.pth` or package symlinks
point to that environment. The remaining pins were installed through uv; SciPy's
official wheel came from the USTC mirror. Source/package details are in
`stable-binary-provenance.json`; successful installation logs are
`copy-stable-binaries-parallel.log` and `install-nwm-stable-final.log`.
Lightning 2.6.5, PyTorch Lightning 2.6.5, WebDataset 1.0.2, BraceExpand 0.1.7
and Tabulate 0.9.0 were added only to import the old CompACT during migration
comparison; the new NWM does not need them. Their installation was constrained
by the production requirements and did not change existing production versions.

Both NWM comparisons passed again in this environment after the final inference
fix, with the same zero tolerances:

| Original implementation | Final stable result | PID / report |
| --- | --- | --- |
| CompACT | 14 cases, 8,372 tensors and 2,744 scalars, maximum difference 0; 126.65 s, exit 0 | 1834924; `stable_nwm_parity/report.json` |
| Meta NWM | 587 tensors and 110 scalars, maximum difference 0; 121.91 s, exit 0 | 1834925; `stable_original_nwm_parity_final/report.json` |

`stable-parity-after-clip-fix-commands.json` records both exact commands. Logs are
`stable-nwm-parity-final.log` and `stable-original-nwm-parity-final.log`.
The final comparison caught an inference regression introduced while restoring
original NWM compatibility: both original implementations disable denoised-latent
clipping, but the added branch had enabled clipping for CompACT. The wrapper now
explicitly preserves `clip_denoised=False` for both modes. The failing evidence
is retained in `stable_nwm_parity_pre_clip_fix/`. Two regression cases exercise
the real diffusion sampler with a predicted latent outside `[-1, 1]`, so the
test fails if clipping is enabled again. The eight related tests in
`nwm/tests/test_run_management.py` passed in the stable environment after the fix
(`nwm-final-regression.log`, exit 0). These include six existing tests and the
two new clipping cases, bringing the final unique NWM test count to **123**,
plus 47 subtests, across the full-suite and targeted regression commands.

All six production GPU training commands also passed in the stable environment:
Stage 1 step 1 → resume step 2, Stage 2 warmup step 1 → joint step 1 → resume
joint step 2, and original NWM step 1 → resume step 2.
Each exercised periodic validation with real DreamSim, preview/curve output,
and best/latest checkpoint saving. `nwm-cli/stable_smoke.py` retains the exact
commands, and `nwm-cli/stable-report.json` records exit 0 for PIDs 1798504,
1813658, 1814932, 1816608, 1818070 and 1824374 respectively. Individual logs are
`nwm-cli/stable-{stage1,stage1-resume,stage2,stage2-resume,original,original-resume}.log`.
After the clipping fix, cached Stage 1 training on real SANPO images and resume
were repeated with online pixel validation, best/latest checkpoints and
preview/curve outputs.
Both exited 0 (PIDs 1844239 and 1845968); `nwm-cli/final-train-report.json`
records the results and `nwm-cli/final-stage1/` contains the fresh run.

Final GPU evaluation used the existing Nav15 Pixel+Action Stage 2 joint-100k
checkpoint through the final `.env` resource `NWM_NAV15_STAGE2_RUN`.
Time and rollout inference each used two real RECON samples and two DDPM steps,
then repeated the command to exercise continuation. Real LPIPS, DreamSim and
FID evaluation completed with finite outputs and also passed continuation.
Planning used one sample, two candidates, top-k 1 and one optimization step,
then passed continuation. All **10 final commands exited 0**, including the
last planning-resume PID 1949053. `nwm-cli/final_eval_smoke.py` retains the
commands, `nwm-cli/final-eval-report.json` records every PID and outcome, and
individual `nwm-cli/final-*.log` files retain command output. Results are in
`nwm-cli/final-eval/pred/evaluation_summary.json` and `nwm-cli/final-plan/`.
These small runs verify the complete workflows, not model quality.

Path setup was checked with both NavAnywhere profiles, normal local paths and
paths containing spaces, followed by CLI dry runs. Ten existing data/model/cache
resources were linked without copying them. The downloader passed a controlled
HTTP Range continuation check and reused an already complete file. The pinned
public release URLs returned HTTP 206 for range requests.

## Repeat the checks

From the new repository root, using the **same interpreter for each old/new pair**:

```bash
/path/to/lam/python tools/compare_original.py --project lam \
  --original /path/to/original/lam_project --output-dir /path/to/artifacts/parity-lam
conda run --no-capture-output -n open-nwm python tools/compare_original.py --project nwm \
  --original /path/to/original/CompACT --output-dir /path/to/artifacts/parity-nwm
conda run --no-capture-output -n open-nwm python tools/compare_original.py --project nwm_original \
  --original /path/to/original/nwm --output-dir /path/to/artifacts/parity-original-nwm
```

The tool selects `lam/` or `nwm/` in this repository as the candidate, starts two
isolated CPU processes, and writes `original.pt`, `candidate.pt`, two logs and
`report.json`. The candidate loads the original initial weights strictly; its
independent initialization is compared too. Default tolerances are zero. A
nonzero exit status means a mismatch or a worker failure. The old checkouts are
needed only for this migration check, never for ordinary training or evaluation.

The CompACT comparison needs its legacy import dependencies in the test
environment: `lightning==2.6.5`, `pytorch-lightning==2.6.5`,
`webdataset==1.0.2`, `braceexpand==0.1.7` and `tabulate==0.9.0`. Install them with
`-c nwm/requirements.txt` to retain the published production versions.

Run each test suite from its own project directory so its modules and config
roots resolve independently:

```bash
cd lam
/path/to/lam/python -m pytest --confcutdir=tests tests -q
```

```bash
cd nwm
conda run --no-capture-output -n open-nwm python -m pytest --confcutdir=tests tests -q
```

From the repository root, run the additional resume test with the LAM interpreter:

```bash
/path/to/lam/python -m pytest --confcutdir=tests tests/test_lam_resume.py -q
```

Install pytest in a dedicated test environment if it is absent. On this machine,
the original LAM interpreter is
`/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/.venv/bin/python`.
The original NWM conda environment is `nwm`; it initially lacked pytest.

These checks establish implementation parity on deterministic small inputs.
The DINO numerical comparison uses precomputed features; the CompACT comparison
uses an identity tokenizer; the original NWM comparison uses a small stochastic
VAE substitute to check posterior sampling/scaling and decoded tensors. The CLI
checks independently exercise the real DINO and LPIPS weights. These checks do
not establish full-dataset metric equality, numerical equality of external
pretrained teacher/VAE implementations, distributed GPU behavior, or convergence
of a complete training run. Record any subsequent real-checkpoint and dataset
evaluation separately with the checkpoint, config, split, seed and result.

## Shared validation split

```bash
conda run -n open-nwm python tools/make_validation_split.py \
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

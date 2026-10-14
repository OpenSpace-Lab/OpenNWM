# Table 2 navigation replay

`reproduce_paper_table2.py` invokes the unchanged archived
`CompACT_a800_original_seed0_v1/planning_eval.py`. It uses EMA weights from
`joint_0100000` for the approved baseline and DINO experiments. It never selects
a checkpoint using test results.

The three Table 2 datasets are RECON, SCAND, and Unitree Go2, each with the
original 100 navigation windows. HuRoN and Rover corrections concern other
evaluations, so these three Table 2 runs can be included in both report views.

The fixed protocol is CEM N=80, K=5, horizon=8, stride=1, repetitions=3,
one optimization step, 250 DDPM steps, LPIPS/AlexNet cost against reconstructed
goals, batch size 1, four loader workers, seed=42, and sample seed=42+index.
Autocast bfloat16, TF32, `torch.compile`, and candidate microbatch=80 remain
unchanged. Metrics use the original evaluator's trajectory coordinates and
EVO settings; no unit conversion, alignment, or scaling is added.

Run from `/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM` with the explicit
Python interpreter:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python tools/reproduce_paper_table2.py cpu-check --model dino --dataset recon
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python tools/reproduce_paper_table2.py run --model baseline --dataset recon --indices 0:1 --gpu 4
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python tools/reproduce_paper_table2.py aggregate --model baseline --dataset recon
```

`prepare` writes provenance and the exact underlying command without starting
GPU work. `cpu-check` strictly loads EMA weights with mmap and checks every
requested sample's shapes and finite values. It does not initialize CUDA.
`run` checks GPU memory before launching, reports working directory, command,
PID, and log path, waits for the evaluator, then reports exit status and log tail.
It resumes complete per-window metric JSONs. `--indices 0:100` is the default;
`--num-shards N --shard-index I` partitions those positions without duplication.
All six model/dataset combinations must finish for a fresh full comparison.

Results live under
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/dino_main_tables_20261007/table2/<model>/<dataset>/mb80/`.
Each shard has isolated copies of split manifests and index audit output.
`protocol.json` prevents incompatible shards from sharing sample metrics.
`shards/<index-hash>/metadata.json` records source/split hashes, checkpoint
size/mtime, parameters, command, environment, state, PID and return code.

`aggregate` writes `aggregate.json` with `complete`, `count`, `missing_indices`
and `means.{ate,rpe_trans,pos_diff_norm,yaw_diff_norm}`. Exit status is 0 only for
100 complete windows, 2 for incomplete results. It resolves original per-sample
files from the archived report (RECON's first 16 are from an earlier directory),
and writes detailed comparisons in `per_sample_archive_comparison.json`.
For baseline, the comparison is a reproduction check; for DINO it is the model
difference against the archived baseline. A complete aggregate alone does not
establish exact reproduction. Partial means are never treated as Table 2 scores.

`verify-subset --model baseline --dataset recon --indices 0,16` is a CPU-only
gate after the two-window pilot. It requires successful shard metadata,
unchanged checkpoint/config/source/split provenance, and exact equality of all
four saved metrics against each window's original archived source. It writes
`subset_verification.json` and exits nonzero on any missing or differing result.
Passing this gate verifies only those two windows.

The prepared full queue is
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/dino_main_tables_20261007/queues/table2_full.json`.
Its 19 sequential jobs verify the pilot, then run, aggregate, and refresh the
report for baseline and DINO on each of the three datasets. GPU jobs wait for
GPU 4 to have at least 40,000 MiB free. Every aggregate requires all 100 windows;
baseline aggregates additionally require all four metrics to equal the archive
on every window. Any failed command or check stops the queue. The pilot's two
completed RECON windows are reused by the full baseline run.

For the user's authorized L20 comparison, the independent queue is
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/dino_main_tables_20261007/queues/table2_l20_full.json`.
It keeps the same 19-job sequence, unchanged evaluation parameters, complete
100-window coverage, source checks, and GPU 4 / 40,000 MiB memory gate. Its pilot
check explicitly passes `--allow-metric-differences`: source, completed-shard,
coverage, and finite-metric validation still must pass, while archive metric
differences are recorded without stopping the remaining model evaluations.
The L20 pilot check writes `l20_subset_verification.json`, separately from the
strict check's `subset_verification.json`, so concurrent checks cannot overwrite
one another. It preserves `all_metrics_exact`, every metric difference, and an
explicit acceptance policy. A completed L20 check with metric differences means
continuation is authorized; it does not mean exact historical reproduction.
Full L20 aggregates retain per-window archive differences and historical
equivalence status, but no longer require exact archive equality to proceed.
The original strict queue is unchanged.

Validate without launching GPU work:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python /file_system/vepfs/algorithm/dujun.nie/code/OpenNWM/tools/reproduce_paper_queue.py --jobs /file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/dino_main_tables_20261007/queues/table2_full.json --validate
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python /file_system/vepfs/algorithm/dujun.nie/code/OpenNWM/tools/reproduce_paper_queue.py --jobs /file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/dino_main_tables_20261007/queues/table2_l20_full.json --validate
```

The L20 queue is already running for this experiment; inspect its state and
live processes before considering any launch. Do not start a second queue.
The queue records subprocess PIDs, commands, logs, completion checks, and exit
statuses in `table2_full.state.json`. Each job has a separate log under
`queues/logs/table2_full_*.log`. Do not infer a successful run from preparation
or validation alone.
The L20 queue uses its own `table2_l20_full.state.json` and
`queues/logs/table2_l20_full_*.log`. Never run both full queues concurrently;
they evaluate the same windows into the same protocol directories.

The original A800 logs report 33,588 MiB maximum allocated memory and about
313–316 seconds per window. The completed L20 pilot (2026-10-07) took
836.3979 seconds for window 0 and approximately 728.6463 seconds for window 16,
with a PyTorch allocation peak of 33,593 MiB and an observed device memory
usage peak of 44,031 MiB. The remaining 598 windows are provisionally estimated
at 121–130 GPU hours (about 5–5.5 days on one L20), excluding group startup and
compile time. This RECON baseline estimate may not represent other datasets
or DINO. Candidate microbatch remains 80.

The strict pilot check exited 1: window 0 matches all four archived metrics,
while window 16 differs. Provenance and coverage checks passed. The authorized
L20 continuation records this difference in a separate verification artifact;
it does not establish exact reproduction. Historical RECON is itself mixed:
windows 0–15 come from an earlier L20 run, and windows 16–99 from the later A800
run. Current evidence does not isolate hardware as the cause of the difference.

Candidate and DDPM noise use stable semantic keys, but VAE posterior sampling
uses the global CUDA RNG. Changing microbatch shape can change VAE samples as
well as floating-point results and CEM ranking. A different microbatch requires
explicit `--accept-microbatch-change`, is stored in a separate directory, and
must not be described as an exact historical replay. Sharding and resume keep
the fixed per-window seed, but historical numerical equivalence still requires
checking the actual outputs; device, environment, and compile state have not
all been independently controlled.

The archived mean RECON/SCAND RPE values are approximately 0.355803 and 0.277794;
the PDF prints 0.35 and 0.27. This is not nearest rounding. Preserve the full
measured values and document the PDF discrepancy rather than forcing equality.

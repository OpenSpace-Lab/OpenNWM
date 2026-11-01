# Go2 EDU + Jetson Deployment Plan

2026-10-14. This plan targets indoor navigation to a location or goal image using Go2 EDU and an onboard Jetson. The Jetson model, memory, JetPack version, and camera setup remain to be confirmed. Deployment and onboard performance have not yet been validated.

## Scope

OpenNWM/NWM predicts images conditioned on observations and candidate actions. Finetuning can adapt these predictions to a robot's camera, motion, and environment. Localization, obstacle handling, trajectory tracking, and robot control require separate components.

| Capability | Available | Still required |
| --- | --- | --- |
| Image/action-conditioned prediction | NWM/OpenNWM, VAE, inference entry point | Live sensors, fixed preprocessing, latency measurements |
| Go2 adaptation | `nwm/conf/two_stage/go2_adapt.yaml`, `nwm/conf/dataset/go2.yaml` | Robot data, independent splits, compatible checkpoint |
| Image-goal planning | Offline candidate sampling/scoring in `nwm/planning_eval.py` | Online loop, feasible candidates, failure handling |
| Closed-loop execution | No Go2 SDK/ROS2 execution node | Localization, control bridge, tracker, obstacle stop, command arbitration |
| Original NWM baseline | Original training recipe via `--mode original` | Separate Go2 finetuning and compatibility checks |

`go2_adapt` defaults to CDiT-B with multiple action encoders and requires a compatible full real-action checkpoint. Original NWM defaults to CDiT-XL without these adapters. Original NWM, LAM, and Stage 1 checkpoints cannot directly replace that initialization. Website videos, offline ATE/RPE, and existing Office-Go2 results do not establish closed-loop performance on a new robot.

## System Design

```text
Location goal ───────────┐
Goal image → retrieval ──┴→ map/topological route → local visual subgoal
                                                       ↓
RGB + synchronized poses → OpenNWM or NWM → candidate scoring/selection
                                                       ↓
                                               trajectory tracker
                                                       ↓
LiDAR/depth + robot state → command gate → SDK2 Move → robot locomotion
```

Establish a ROS2 localization/Nav2 baseline first. Convert known locations to map coordinates. Retrieve image goals from mapped keyframes, reject low-confidence matches, and choose visible local subgoals. Evaluate OpenNWM/NWM initially in shadow mode, recording candidate scores before enabling local selection. Run the models separately for comparison.

Required modules include a robot bridge, recording converter, model server, local planner, tracker, and command gate. These are proposed components; deployment commands are not yet provided.

## 1. Software and Sensors

- Confirm Jetson hardware, memory, JetPack/CUDA, power mode, storage, temperature, and EDU firmware interfaces. Install compatible aarch64 PyTorch/torchvision; server CUDA wheels from the root README are not a Jetson installation recipe.
- Match ROS2 to JetPack. The referenced Unitree ROS2 setup uses Ubuntu 22.04 and Humble/CycloneDDS. Check networking, DDS, and clock synchronization on the robot.
- Record RGB, camera intrinsics/distortion, and sensor extrinsics. Match training preprocessing and retain realistic pitch, roll, and vibration in training data.
- Combine robot pose/IMU with LiDAR or visual SLAM. Robot state positions alone do not provide drift-free map localization. Fix `map → odom → base_link → camera/lidar` transforms.
- Use high-level motion control and retain factory locomotion. Only one component should own motion commands at a time.

| Function | Interface |
| --- | --- |
| Velocity and stop | [`SportClient.Move(vx, vy, vyaw)` / `StopMove()`](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/sport/sport_client.py) |
| Front camera | [`VideoClient.GetImageSample()` example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/front_camera/camera_opencv.py) |
| ROS2 state, point clouds, control | [`sportmodestate`, `/utlidar/cloud`, `/api/sport/request`](https://github.com/unitreerobotics/unitree_ros2) |

Verify topics, firmware behavior, image timestamps, and frame rate on hardware. A control bridge must convert Nav2 `cmd_vel` into Unitree high-level requests.

## 2. Data and Action Definitions

Start with roughly 1–3 hours of teleoperation as an engineering pilot, then add data for observed failures. Cover straight motion, turns, rotations, starts/stops, corridors, doorways, lighting, and traversable obstacle layouts. Preserve raw timestamps and resample to the current Go2 recipe's 4 Hz offline.

Record RGB, measured poses, IMU, point clouds/depth, commanded velocities, collisions, and interventions. **Derive action labels from measured poses.** Store commands separately so slip and control lag are not mislabeled as completed motion.

The current recipe uses 224×224 images and four context frames, spanning 0.75 seconds at 4 Hz:

```text
$NWM_GO2_ROOT/go2/episode_001/
    0.jpg, 1.jpg, ...
    traj_data.pkl                 # position: float array [N,2/3]; yaw: [N], radians
$NWM_SPLIT_ROOT/go2/train/traj_names.txt
$NWM_SPLIT_ROOT/go2/val/traj_names.txt
$NWM_SPLIT_ROOT/go2/test/traj_names.txt
```

Separate complete episodes, routes, or collection days into train/val/test, for example 70/15/15. Keep timestamps, calibration, and intervention flags in sidecars. Remove windows crossing timestamp gaps, localization jumps, or episode boundaries. `len_traj_pred=64` requires sufficient continuous footage for a 16-second future at 4 Hz.

Real actions represent planar relative poses in the reference observation frame:

```text
local_xy = R(-yaw_t) @ (p_g - p_t)
model_xy = local_xy / metric_waypoint_spacing
model_yaw = wrap(yaw_g - yaw_t)
```

Reuse the complete handling in `datasets.py`, `misc.py`, and the action adapters. Training, rollout, and planning paths differ in incremental/cumulative poses and normalization; avoid applying normalization twice. Check straight, left, right, and ±π-crossing trajectories. Define the reference frame and apply camera/body extrinsics consistently.

The default spacing `0.07610250112606785` belongs to the original Go2 data. Estimate any replacement from the training split, freeze it, and use it consistently in training, inference, evaluation, and metric trajectory reconstruction. Record and validate any change from the pretrained scale.

Model poses cannot be passed directly to `Move`. The tracker converts trajectories in meters/radians into m/s and rad/s using timing and motion constraints. Rotate local increments before accumulation. A general Go2 rosbag converter is not included; implement conversion and coordinate/clock checks before finetuning.

## 3. Server Finetuning

Start from a compatible OpenNWM real-action checkpoint. The Go2 recipe uses 200 adapter-warmup steps and 800 joint steps, adapter/backbone learning rates of `1e-5/1e-6`, and batch size 8. Select training duration and checkpoints using independent validation.

After preparing data, run this template from the repository root. Replace paths, checkpoint, scale, and GPU selection for the target system:

```bash
source .env
unset NWM_PYTHON
export NWM_GO2_ROOT=/data/go2_navigation/data
export NWM_SPLIT_ROOT=/data/go2_navigation/splits
go2_init_checkpoint=/absolute/path/to/compatible_real_action_checkpoint.pth.tar
go2_run=/data/go2_navigation/runs/opennwm_adapt_v1
# Replace with the scale computed from and frozen for the training split.
go2_spacing=0.07610250112606785

nvidia-smi
df -h /data
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n open-nwm python run.py nwm train \
  --mode go2_adapt --gpus 1 --run-dir "$go2_run" \
  finetune.init_checkpoint="$go2_init_checkpoint" \
  dataset.datasets.go2.metric_waypoint_spacing="$go2_spacing" \
  training.eval_batch_size=2 training.eval_num_batches=1
```

The historical training field `dataset.datasets.go2.test` points to `go2/val`. Final testing requires explicit Go2 paths, test split, and sampling indices in `evaluation_datasets`; no automatic val-to-test switch occurs. The existing `unitree_go2` evaluation entry uses previously converted data, not new robot recordings. Small validation sets must still contain enough valid windows.

Record the working directory, command, GPU, log path, and background PID. Monitor logs and exit status. Store data, VAE caches, checkpoints, and logs outside Git.

An original NWM comparison needs a separate adaptation path retaining its legacy architecture/action encoding. Distinguish weight initialization from full training-state resume and evaluate on the same splits. `--mode original` alone does not implement Go2 finetuning. `tools/finetune_lusnar.py` illustrates strict configuration-based loading and adapter/legacy interfaces, but its data protocol is specific to LuSNAR.

## 4. Jetson Performance

First run batch-size-1 PyTorch inference with complete preprocessing. Measure peak shared memory, temperature/throttling, and end-to-end p50/p95 latency with `tegrastats`. Parameter counts, TOPS, or one denoiser-step timing do not establish navigation throughput.

Offline planning defaults to 80 candidates, three repeats, and 250 diffusion steps, plus VAE and image scoring. The [2026-10-07 L20 baseline pilot](tools/reproduce_paper_table2.md) took about 836 and 729 seconds for two RECON planning windows, with 33,593 MiB peak PyTorch allocation and 44,031 MiB observed device memory. These are historical offline measurements, not Jetson results or optimized lower bounds; the current planner requires substantial adaptation for onboard use.

Test optimizations in this order, rechecking prediction and navigation quality:

1. Cache observation VAE encodings, keep models resident, and remove file output/plotting/metrics from the control loop.
2. Filter physically feasible candidates before model scoring; reduce candidate count, repeats, and horizon.
3. Validate lower precision and fewer sampling steps. Consider fixed-shape VAE/denoiser TensorRT exports separately from the Python planner.
4. If latency remains high, score at a lower frequency or distill a lightweight policy from offline model supervision.

Initial design targets could be 1–2 Hz model updates and 20–50 Hz tracking/control, subject to measurement and SDK limits. Plans need observation timestamps, coordinate frames, and expiry times. Compensate for intervening pose changes or discard invalid plans; slow or stop when replanning cannot keep up. Jetson capacity and model throughput remain unknown until hardware is specified and tested.

## 5. Validation

| Stage | Acceptance checks |
| --- | --- |
| Control/localization | Correct low-speed motion axes, calibrated poses, stop on command timeout |
| Navigation baseline | Straight routes, turns, doorways; record success, endpoint error, time, interventions |
| Offline model | Held-out routes, pre/post-finetuning prediction and planning, metric error, wrong-action sensitivity |
| Shadow mode | Candidate feasibility, goal progress, latency, plan validity |
| Low-speed closed loop | Open routes, then corridors/doorways; independent obstacle stop and manual emergency stop |
| Model comparison | Same tasks for Nav2, OpenNWM before/after adaptation, and original NWM before/after adaptation |

An initial evaluation can use 20–30 indoor tasks with at least three repeats each. Treat ≥90% arrival and ≤0.5 m endpoint error as provisional engineering targets, adjusted to map/localization accuracy. Also report collisions, interventions, completion time, path length, p95 latency, and memory.

Stopping distance must cover sensing/model/control latency, measured braking distance, and localization margin. Validate the baseline, data conversion, server adaptation, Jetson shadow mode, and low-speed closed loop in that order.

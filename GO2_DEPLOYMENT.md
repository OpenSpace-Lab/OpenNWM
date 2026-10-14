# Go2 EDU + Jetson：室内导航部署与微调方案

2026-10-14。目标硬件为 Go2 EDU 与机载 Jetson，首阶段在已知室内区域到达指定位置或目标图片。Jetson 型号、内存、JetPack 和相机配置尚待确认。本文是基于现有代码的实施方案，不是已完成的实机部署或性能报告。

## 可行性与现有边界

可以开展原型验证。OpenNWM/NWM 是动作条件世界模型：根据观测与候选动作预测未来图像，再用于选择轨迹。微调改善本机相机、运动与环境的匹配，但不会自动提供定位、避障、轨迹跟踪或机器人控制。

| 能力 | 仓库现状 | 实机还需要什么 |
| --- | --- | --- |
| 图像与真实动作条件预测 | 已有 NWM/OpenNWM 模型、VAE、推理入口 | 接入实时传感器、固定预处理、测延迟 |
| Go2 适配 | `nwm/conf/two_stage/go2_adapt.yaml`、`nwm/conf/dataset/go2.yaml` | 本机采集数据、独立 split、兼容 checkpoint |
| 图像目标规划 | `nwm/planning_eval.py` 离线候选采样与评分 | 在线循环、候选可行性约束、失效处理 |
| 机器人闭环执行 | 未提供 Go2 SDK/ROS2 执行节点 | 定位、控制桥、跟踪器、近障停车、指令仲裁 |
| 原始 NWM 对照 | `--mode original` 保留原始训练配方 | 独立的 Go2 微调适配与兼容性验证 |

`go2_adapt` 默认是 CDiT-B 与多类型动作编码配置，要求完整且结构兼容的真实动作权重。原始 NWM 模式默认 CDiT-XL、关闭该适配器；不能把原始 NWM checkpoint 直接替换进同一条 Go2 命令。LAM/Stage 1 checkpoint 也不能直接替代完成真实动作训练的模型。

网站预测视频、离线 ATE/RPE 和现有 Office-Go2 数据结果均不能作为这台机器的闭环成功率。

## 建议的系统

```text
位置目标 ───────────┐
目标图片 → 地点检索 ─┴→ 地图/拓扑路径 → 近处子目标
                                          ↓
RGB + 同步位姿 → OpenNWM 或 NWM → 候选轨迹评分/选择
                                          ↓
                                  局部轨迹跟踪器
                                          ↓
LiDAR/深度 + 状态 → 指令仲裁、限速、超时停车 → SDK2 Move → 原厂运动控制
```

先建立 ROS2 + SLAM/定位 + Nav2 的可复现基线；如果已经有稳定地图，就先做重定位与路径跟踪。已知位置转换为地图坐标目标；目标图片先从采集的关键帧库检索地点，关联其地图位姿，并设置低置信度拒绝规则。地图层生成相机可观察的近处子目标，避免直接用隔墙的远处图片优化局部轨迹。

OpenNWM/NWM 先作为只记录输出的旁路模块，对同一批候选轨迹评分；离线与旁路验证通过后，接入局部选择。两模型分开运行并比较，不必同时常驻 Jetson。如果研究目标是完全由世界模型完成局部规划，还需要实现和验证在线候选生成、物理约束与滚动重规划。

最小新增模块可以划分为 `go2_bridge`、`record_convert`、`world_model_server`、`local_planner`、`trajectory_tracker` 和 `command_gate`。这些是待开发模块名称，当前仓库不存在对应的部署命令。

## 1. 先打通机载软件和传感器

- 确认 Jetson 型号/内存、JetPack/CUDA、功耗模式、温度、可用存储与 EDU 固件接口。使用与 JetPack 匹配的 aarch64 PyTorch/torchvision；根 README 中服务器用的 CUDA wheel 安装命令不能直接照搬到 Jetson。
- 官方 ROS2 仓库推荐 Ubuntu 22.04 + Humble/CycloneDDS；具体选择须与 JetPack 系统匹配。确认网卡、DDS 配置和时钟同步，再建立 ROS2 节点。
- 获取 RGB、相机内参/畸变和安装外参；图像经与训练一致的去畸变、裁剪和缩放。腿式运动的俯仰、横滚和振动应保留在本机训练样本中。
- 记录状态中的 position/IMU/yaw，结合 LiDAR 或视觉 SLAM 提供定位。原厂状态位置不是天然无漂移的地图定位；固定 `map → odom → base_link → camera/lidar` 变换关系。
- 使用高层运动接口，保留原厂站立/步态控制。保证同一时刻只有一个模块拥有运动指令发布权。

已核实的官方接口：

| 用途 | 接口与来源 |
| --- | --- |
| 机体速度与停车 | [`SportClient.Move(vx, vy, vyaw)` / `StopMove()`](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/sport/sport_client.py) |
| 前视相机 | [`VideoClient.GetImageSample()` 官方 OpenCV 示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/front_camera/camera_opencv.py) |
| ROS2 状态、点云和控制 | [`sportmodestate`、`/utlidar/cloud`、`/api/sport/request`](https://github.com/unitreerobotics/unitree_ros2) |

主题名称、固件发布内容、图像时间戳和实际帧率仍须在实机核对。Nav2 的 `cmd_vel` 要通过控制桥转换为 Unitree 高层请求。

## 2. 采集本机数据并固定动作定义

先遥操作采集约 1–3 小时作为首轮工程试验预算，不是保证收敛的数据量。覆盖直行、左右转、原地旋转、启停、走廊、门口、不同光照与可通行障碍布局；随后根据失败场景补数据。尽量保留传感器原始频率和时间戳，离线对齐/重采样为当前 Go2 配方的 4 Hz。

同时记录 RGB、带时间戳的实际位姿、IMU、点云/深度、下发速度、碰撞或人工接管事件。**动作标签由实际位姿生成**，控制指令另外保存用于诊断，避免将打滑或控制滞后误标为已实现运动。

当前配置使用 224×224 图像与 4 帧上下文；4 Hz 下首尾跨度为 0.75 秒。训练数据组织为：

```text
$NWM_GO2_ROOT/go2/episode_001/
    0.jpg, 1.jpg, ...
    traj_data.pkl                 # position: float array [N,2/3], yaw: [N], radians
$NWM_SPLIT_ROOT/go2/train/traj_names.txt
$NWM_SPLIT_ROOT/go2/val/traj_names.txt
$NWM_SPLIT_ROOT/go2/test/traj_names.txt
```

按完整采集回合/路线/采集日期隔离 train、val、test，例如 70/15/15；不要随机拆相邻帧。时间戳、标定和接管标记另存 sidecar，`traj_data.pkl` 保持 loader 所需的数值数组。删除跨时间断点、定位跳变和不同回合拼接的窗口。当前训练需要足够长的连续片段：`len_traj_pred=64`，4 Hz 对应未来 16 秒；不是任意短片段都能被采样。

模型真实动作语义是参考观测坐标下的平面相对位姿。对于当前位置 `(p_t, yaw_t)` 与候选时刻 `(p_g, yaw_g)`：

```text
local_xy = R(-yaw_t) @ (p_g - p_t)
model_xy = local_xy / metric_waypoint_spacing
model_yaw = wrap(yaw_g - yaw_t)
```

重用 `datasets.py`、`misc.py` 和动作适配器的完整处理逻辑；不同训练、rollout 和 planner 路径还涉及增量、累计位姿和归一化，不能额外重复归一化。用直行、左转、右转与跨 ±π 的已知轨迹检查坐标轴和角度。相机坐标系与机体坐标系不同，必须明确选择并使用标定外参。

配置中的 `0.07610250112606785` 是原 Go2 数据的 waypoint spacing，不能当作所有 Go2 的常数。选择本机尺度时，只用训练集估计并冻结它；训练、推理、评测与米制轨迹还原必须一致。切换预训练尺度需要作为适配方案记录和验证。

模型的位姿数值不能直接传给 `Move`。跟踪器在当前机体坐标下跟踪以米和弧度表示的轨迹，根据时间间隔、控制周期和运动约束计算 m/s、rad/s；转弯中的连续局部增量需要先旋转再累计。

本仓库目前没有通用的 Go2 rosbag 转上述数据格式的工具；需新增转换器和坐标/时钟验证，再执行微调。

## 3. 在训练服务器微调，再把权重部署到 Jetson

第一轮用 OpenNWM 的结构兼容、已完成真实动作训练的 checkpoint。已有 Go2 配方是 200 步 adapter warmup + 800 步联合微调，adapter/backbone 学习率分别为 `1e-5/1e-6`，batch 为 8。这是验证流程的起点；按独立 val 决定训练时长和 checkpoint，不保证 1,000 步就能导航。

下面是**数据准备完成后的训练服务器命令模板**，本次没有启动训练。当前机器可用环境名为 `nwm`；新机器按自己的环境名替换。加载 `.env` 后取消解释器覆盖，确保下方明确选择的 Conda 环境贯穿启动。NAS 路径为本机建议位置，新机器自行设置挂载路径。

```bash
cd /file_system/vepfs/algorithm/dujun.nie/code/OpenNWM
source .env
unset NWM_PYTHON
export NWM_GO2_ROOT=/file_system/nas/algorithm/dujun.nie/go2_navigation/data
export NWM_SPLIT_ROOT=/file_system/nas/algorithm/dujun.nie/go2_navigation/splits
# 必须替换为存在且结构兼容的完整 real-action checkpoint。
go2_init_checkpoint=/absolute/path/to/compatible_real_action_checkpoint.pth.tar
go2_run=/file_system/nas/algorithm/dujun.nie/go2_navigation/runs/opennwm_adapt_v1
# 设为本次训练集计算并冻结的尺度；此处仅展示旧数据的默认值。
go2_spacing=0.07610250112606785

nvidia-smi
df -h /file_system/nas/algorithm/dujun.nie
# 根据上一步选择空闲 GPU；下面仅示例 GPU 0，不应覆盖其他实验。
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nwm python run.py nwm train \
  --mode go2_adapt --gpus 1 --run-dir "$go2_run" \
  finetune.init_checkpoint="$go2_init_checkpoint" \
  dataset.datasets.go2.metric_waypoint_spacing="$go2_spacing" \
  training.eval_batch_size=2 training.eval_num_batches=1
```

训练默认将历史字段 `dataset.datasets.go2.test` 指向 `go2/val`。最终测试须显式配置 `evaluation_datasets` 的 Go2 数据目录、`go2/test` 路径与对应采样索引；入口不会自动从 val 切到 test。现有独立评测项名为 `unitree_go2`，对应既有转换数据，不等于本次新采集数据。微调校验允许本地指标与 val 评估；仍要求完整兼容权重或有效恢复 checkpoint。小验证集还需保证有效窗口数足够。

长任务启动前记录工作目录、完整命令、GPU、日志路径；后台运行时记录 PID，监控日志和退出状态。数据、VAE 缓存、模型 checkpoint 和日志放在 NAS，不提交 Git。

原始 NWM 对照应另建保留 legacy 架构与动作编码的微调入口，明确只初始化权重还是恢复优化器/训练计数，再对同一 split 验证。`--mode original` 是训练配方选择，不能据此声称原始 NWM 的 Go2 微调链已打通。两模型对照同时报告模型大小、采样配置和耗时，避免混淆架构差异与预训练收益。

可以参考 `tools/finetune_lusnar.py` 的 `load_model` / `model_kwargs`：它按源配置严格加载，并区分适配器与 legacy 动作接口；但该脚本的数据读取与训练协议针对 LuSNAR，尚不能直接训练 Go2。

## 4. 先测 Jetson 性能，再确定规划频率

机载端先以 PyTorch 跑通 batch=1 和完整预处理，测峰值内存、温度/降频、端到端 p50/p95 延迟。使用 `tegrastats` 看共享内存和功耗；仅报模型参数量、TOPS 或去噪器单步时间不足以说明可实时导航。

当前离线规划默认 `num_samples=80`、`num_repeat_eval=3`，默认扩散评估 250 步，还包含 VAE 和图像评分。因此不能把 `run.py nwm plan` 直接包装成 10 Hz 实机规划器。

已有[离线复现实验记录](tools/reproduce_paper_table2.md)提供了具体量级：2026-10-07 的 L20 RECON baseline pilot，两个规划窗口分别约 836 秒和 729 秒，PyTorch 分配峰值 33,593 MiB，观测到的设备显存峰值 44,031 MiB，候选 microbatch 为 80。这些是特定离线协议的历史测量，不是本次 Jetson 实测，也不是优化后的性能下界；但表明当前完整规划协议需要显著改造，不能仅靠小规模微调后原样机载运行。

按以下次序探索，所有改动均需重新检查预测/导航质量：

1. 缓存观测 VAE 编码、复用常驻模型、移除文件写入/绘图和离线指标计算。
2. 在已满足碰撞与运动约束的候选中先筛选，再用世界模型评分；减少候选数、重复次数、规划时域。
3. 验证合适的低精度和少步采样方案；可试验导出稳定形状的 VAE/去噪器到 TensorRT，不假设整个 Python planner 可直接导出。
4. 若端到端延迟仍过大，让世界模型低频评分/只在关键点调用；或者用其离线生成的监督蒸馏轻量策略，再单独验证策略。

首轮可把模型层 1–2 Hz、局部跟踪/指令层 20–50 Hz 作为待测设计目标，具体频率受型号、模型、候选数和 SDK 约束。若无法及时重规划，就降低速度或停车等待，不能重复执行过期计划。每份计划应携带观测时间戳、坐标系与有效期；新旧位姿间的变化要补偿，超限则丢弃。

Jetson 未明确型号前，不承诺 NWM XL 或完整多候选扩散规划的帧率、内存占用或可用性。训练优先放服务器，Jetson 专注推理与执行。

## 5. 分阶段验收

| 阶段 | 验收内容 |
| --- | --- |
| 控制与定位 | 手动低速前进/左右转方向正确；定位与外参一致；失联/指令超时可停车 |
| 传统导航基线 | 同一地图上直行、转弯、穿门到点，保存成功率、终点误差、耗时与接管次数 |
| 离线模型 | 独立路线比较微调前后预测与规划，检查实际米制误差和错误动作敏感性 |
| 旁路运行 | 模型只输出候选与评分；核对可通行性、目标进展、延迟及计划有效期 |
| 低速闭环 | 开阔路线起步，随后走廊/门口；独立近障停车与人工急停始终可用 |
| 对照试验 | Nav2 基线、OpenNWM 微调前/后、原始 NWM 微调前/后用同一任务集合比较 |

可先固定 20–30 个室内任务，每个重复至少 3 次；将 ≥90% 到达率、≤0.5 m 终点误差作为初始工程验收目标，并根据地图/定位精度调整。另报碰撞率、人工接管、完成时间、轨迹长度、p95 延迟与内存，不能只用好看的预测视频代替导航结果。

近障停止距离应覆盖传感器/模型/控制延迟内的行进距离、制动距离和定位余量；不能只依赖世界模型生成的图像判断安全。先实测低速制动，再设速度和停止阈值。

## 推荐实施顺序

先完成 EDU/Jetson 接口、定位和到点基线；接着做本机记录转换与 OpenNWM 小规模微调；然后在 Jetson 上测端到端推理和旁路评分；最后接入低速闭环，并补原始 NWM 的独立微调对照。每阶段通过验收再进入下一阶段。最终能否在指定机载配置上达到目标速度与成功率，需要这些实测结果回答。

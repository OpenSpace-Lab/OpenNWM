> 当前实时导航使用固定 15 候选：静止直接以当前真实图像与目标计算 LPIPS，14 个运动候选批量推理；默认 batch size 14（兼容旧参数 15）。所有运动候选统一使用 0.5 秒条件（k=0.015625），默认加载随代码保存的合格离线标定。执行确认只需输入 MOVE，动作结束后显示实际图像与目标的 LPIPS。请先阅读 [离线标定与导航](OFFLINE_CALIBRATION.md)。下文旧的 7/9/45 候选和在线补偿导航示例不再适用于当前导航入口；其他离线推理入口不受影响.

# Jetson 离线推理

目标环境：Go2 机载 Orin NX 16GB，L4T 35.3.1、CUDA 11.4、cuDNN 8.6、Python 3.8。
环境位于 `/home/unitree/ndj/envs/opennwm`，代码位于 `/home/unitree/ndj/code/OpenNWM`，权重位于 `/home/unitree/ndj/models`。
只在 `deployment` 分支维护部署改动。

## 使用

```bash
source /home/unitree/ndj/envs/opennwm/bin/activate
cd /home/unitree/ndj/code/OpenNWM
python -B deploy/jetson_infer.py \
  --image website/assets/playground/garden-recon/initial.webp \
  --goal website/assets/playground/garden-recon/initial.webp
```

该命令从文件读取 RGB，加载 OpenNWM Stage 2 EMA 和 SD-VAE，执行一次 250 步 DDPM 预测，并计算预测图与目标图的 LPIPS/AlexNet 距离。同一张初始图重复为四帧上下文。它不导入 Unitree SDK，不连接 ROS/DDS，不读取实时摄像头，不执行运动控制。目标图与初始图相同只用于验证计算流程，不是导航成功指标。

默认只打印结果，不产生图像或日志。加 `--output /home/unitree/ndj/outputs/prediction.png` 可保存预测，已有文件不会覆盖。
加 `--model nwm-b` 使用原始 NWM-B；加 `--steps 25` 可做较快的连通性验证，但采样步数会影响生成质量。
输出中的 `inference_seconds` 和 CUDA 峰值分配量仅对应图像预测，不包括权重加载和 LPIPS；Jetson 共享内存的系统总占用也会更高。

`--delta X Y YAW` 是供模型预测用的假设相对位姿，XY 使用训练数据的 waypoint-spacing 单位，角度为弧度，默认全零。不是米、速度或机器狗指令。OpenNWM 使用模型内归一化，NWM-B 使用原始模型的平移 min/max 归一化。
`--frames-ahead` 默认 1，采用模型 4 fps 的时间约定。需要可靠预测时应提供真实历史上下文，并先验证机载摄像头数据的分布与动作尺度；此入口只验证单图冷启动推理。

## DMD 学生权重推理

支持 ModelScope 发布的 `LittleBoss/OpenNWM-DMD` 和 `LittleBoss/NWM-DMD`。
在 `--model-root` 下分别放置 `OpenNWM-DMD/{student.pt,config.yaml}` 和
`NWM-DMD/{student.pt,config.yaml}`，共享已有的 `sd-vae-ft-ema/`；
导航评分仍需要 `planning/` 下的 LPIPS/AlexNet 权重。
本机模型根目录为 `/home/hongyu/ndj/models`，机载可使用原来的
`/home/unitree/ndj/models`。加载学生时使用发布的配置及严格 `ema` 参数匹配，
不会读取教师 checkpoint，也不依赖教师资产清单。

```bash
# 在仓库根目录、已安装部署依赖的环境中运行；只读本地图片，不连接机器人。
python -B deploy/jetson_infer.py \
  --model opennwm-dmd --model-root /home/hongyu/ndj/models \
  --image website/assets/playground/garden-recon/initial.webp \
  --output /tmp/opennwm-dmd-prediction.png

python -B deploy/jetson_infer.py \
  --model nwm-dmd --model-root /home/hongyu/ndj/models \
  --image website/assets/playground/garden-recon/initial.webp \
  --output /tmp/nwm-dmd-prediction.png
```

`--model opennwm-dmd` / `--model nwm-dmd` 自动选择 `--sampler dmd --steps 4`。
也可给原来的 `--model opennwm` / `--model nwm-b` 加 `--sampler dmd` 来加载对应学生。
显式指定其他步数、DDPM/DDIM 采样器或 `--ddim-eta` 与 DMD 模型混用会报错。
原教师模型的默认采样器、默认步数及显式 DDIM 参数保持原意。

导航 `jetson_navigate.py preview/navigate` 支持相同两个模型名；将已有命令中的
`--model` 改为 `opennwm-dmd` 或 `nwm-dmd`，删除旧 `--sampler ddim --steps 10`
（或显式改为 `--sampler dmd --steps 4`）。MOVE/arrived/q、运动限位及人工确认不变。
`jetson_video.py` 和 `jetson_episode.py infer` 使用 `--sampler dmd` 同时评估两个学生，
其输出模型键仍是 `opennwm` / `nwm-b`，清单记录 `sampler: dmd`、实际学生路径及文件哈希。
`jetson_select.py` 可继续读取该视频清单。

`deploy/dmd_sampler.py` 对齐发布的 `dmd2.py` 四步数学过程：
原始训练索引 `[999,749,499,249]`，从 epsilon 通道还原未裁剪的 x0，
前三步之间重新加独立噪声，末步直接返回 x0；方差通道不参与采样。
完整 1000 步训练噪声表仅用于计算系数，模型实际调用四次。
固定 sample key/seed 的初始噪声及重加噪声在候选批次间复用，改变 batch size 不改变噪声语义。
VAE 使用 posterior mode；NWM-DMD 外部归一化 XY、保留 yaw，
OpenNWM-DMD 将原始动作交给模型内 motion adapter。沿用既有 rel_t 和动作单位。

这些发布权重属于短程 DMD2 pilot；接通推理不代表已验证真实导航质量。
部署验证可运行：

```bash
python -m unittest discover -s deploy/tests -v
```

## 教师模型的可选少步采样器

四个推理入口 `jetson_infer.py`、`jetson_video.py`、`jetson_episode.py infer`、`jetson_navigate.py preview/navigate` 均接受 `--sampler ddpm|dpmpp_2m|ddim`，步数继续使用已有 `--steps`，不增加第二个步数参数。不传 `--sampler` 仍调用原 DDPM；单图默认 250 步，视频、episode、导航默认 50 步，其他显式步数保持原意。prepare/probe 不采样，拒绝加速采样器选项。

`dpmpp_2m` 使用 DPM-Solver++ 二阶多步 midpoint，`lower_order_final=True`、dynamic thresholding 关闭，终点 sigma 为 0。`ddim` 默认 eta=0；仅 DDIM 接受 `--ddim-eta`，范围 [0,1]。给 DDPM/DPM-Solver++ 传入该参数会报错。新路径要求 2 到训练扩散总步数之间的整数步数；部署 CLI 另保留原有 2..1000 的上限。不会自动退回 DDPM。

模型仍在 SD-VAE latent 空间去噪，保留 checkpoint 的 epsilon/x0 类型和 sigma 通道定义。原 `SpacedDiffusion.p_mean_variance` 按实际 C 拆分 2C 输出、转换到 x0，再将 C 通道 x0 交给 DPM-Solver++，因此求解器的 `prediction_type=sample` 不代表修改了 epsilon checkpoint。新采样表从训练的完整 beta schedule 重新构造，覆盖末端训练 timestep 到 0，再到干净样本；不是截取原 50 步循环。当前训练索引映射不做额外缩放。DPM-Solver++ 使用完整训练 betas 和显式映射时间表，每次独立生成创建新求解器。

实际入口保持 `clip_denoised=False`，不裁剪 latent x0；VAE scaling 0.18215、解码和 RGB 后处理仍使用原代码。历史、动作、rel_t、批次、候选和 rollout 的处理均未变，目标 RGB 仍用于原有 LPIPS 评分。当前推理入口没有 CFG 或图像 inpainting 接口；不新增这些功能。底层 DDIM 保留原 condition_score 梯度条件语义；DPM-Solver++ 对 `cond_fn` 明确报错。未识别的 diffusion factory、v-prediction 或非整数 respacing 在新路径中拒绝使用，旧 DDPM 不受影响。

继续使用既有 sample key、seed、noise stream 和初始噪声生成方法。确定性采样器不消费逐步随机噪声；随机 DDIM 使用传入的逐步噪声。固定 seed 不保证不同步数/采样器结果或底层形状相关随机张量逐位一致。新增逻辑不改变旧 DDPM 的随机调用顺序。使用项目已声明的 Diffusers（Jetson 0.30.3、桌面 0.35.1），没有安装或升级依赖；新调度器只在选择 DPM-Solver++ 时导入。

以下是可复用的**离线单图**对比命令。将占位符替换成现有路径，在仓库根目录执行。`--model-root` 必须包含入口原有的 `OpenNWM/configs/opennwm-nav15-stage2.yaml`、`OpenNWM/opennwm-nav15-stage2-joint100k.pth.tar`、`sd-vae-ft-ema/` 和目标评分权重；该入口没有单独的 `--checkpoint` 参数。

```bash
NWM_TEST_PYTHON='/绝对路径/现有NWM环境/bin/python'
MODEL_ROOT='/绝对路径/模型根目录'
INPUT_IMAGE='/绝对路径/输入图像.png'
GOAL_IMAGE='/绝对路径/目标图像.png'
COMPARE_OUTPUT='/绝对路径/新的比较目录'
common=(deploy/jetson_infer.py --model opennwm --model-root "$MODEL_ROOT"
        --image "$INPUT_IMAGE" --goal "$GOAL_IMAGE"
        --frames-ahead 4 --delta 0.5 0 0 --seed 0)

# 原 DDPM 50 步（省略 --sampler 也仍为 DDPM）
"$NWM_TEST_PYTHON" -B "${common[@]}" --steps 50 --output "$COMPARE_OUTPUT/ddpm-50.png"
"$NWM_TEST_PYTHON" -B "${common[@]}" --sampler dpmpp_2m --steps 20 --output "$COMPARE_OUTPUT/dpmpp-20.png"
"$NWM_TEST_PYTHON" -B "${common[@]}" --sampler dpmpp_2m --steps 15 --output "$COMPARE_OUTPUT/dpmpp-15.png"
"$NWM_TEST_PYTHON" -B "${common[@]}" --sampler dpmpp_2m --steps 10 --output "$COMPARE_OUTPUT/dpmpp-10.png"
"$NWM_TEST_PYTHON" -B "${common[@]}" --sampler ddim --steps 20 --output "$COMPARE_OUTPUT/ddim-20.png"
"$NWM_TEST_PYTHON" -B "${common[@]}" --sampler ddim --steps 10 --output "$COMPARE_OUTPUT/ddim-10.png"
```

六条命令固定同一 OpenNWM EMA、VAE、输入/目标、动作、seed 和现有精度；单图重复为相同四帧历史，batch=1、候选=1、输出未来帧=1，预测跨度为 4/4=1 秒，分辨率由同一 checkpoint 配置决定（当前为 224×224）。扩散步数不影响这些设置。所有输出文件必须尚不存在。此入口是冷启动单图比较，不代表真实历史、多候选导航质量或端到端速度已验证。

### 机载离线对比（2026-10-19）

已在 NVIDIA Orin NX 16GB、15W 模式、CUDA 11.4/cuDNN 8.6、现有 BF16 autocast 下完成六组采样器对比。使用同一 OpenNWM EMA、VAE、seed=0、224×224、四张真实历史帧和固定 1 秒模型时间条件；核对了所有场景及 batch 下不同步数的初始噪声完全相同。每种配置/batch 先预热一次，再在三个已有场景各测一次，共 36 次正式调用、144 张预测。batch=1 固定 forward_slow；batch=7 使用原七个固定候选，不代表 45 候选完整导航测试。

| 采样设置 | 单样本中位秒数 | 七候选整批中位秒数 | 整批加速比 | 平均最佳目标 LPIPS↓ | 与 DDPM 选点相同 |
|---|---:|---:|---:|---:|---:|
| DDPM 50 | 9.08 | 62.16 | 1.00× | 0.6324 | 3/3 |
| DPM-Solver++ 20 | 3.88 | 26.42 | 2.35× | 0.6698 | 1/3 |
| DPM-Solver++ 15 | 3.00 | 20.48 | 3.04× | 0.6796 | 1/3 |
| DPM-Solver++ 10 | 2.12 | 14.53 | 4.28× | 0.6854 | 1/3 |
| DDIM 20 | 3.89 | 26.45 | 2.35× | 0.6392 | 0/3 |
| DDIM 10 | 2.13 | 14.54 | 4.27× | 0.6409 | 0/3 |

计时包含 noise/conditioning、去噪和 VAE 解码，用单调时钟并同步 CUDA；不含权重加载、已缓存的历史 VAE 编码、LPIPS 或磁盘写入。LPIPS 另计，七候选加入评分后的中位时间约增加 0.033 秒。模型/VAE/LPIPS 加载实测约 7.04 秒；四帧历史编码首次约 3.38 秒，之后两个场景约 0.70 秒。这些是本次运行条件下的耗时，不是冷开机启动或完整导航延迟。

各采样器测得 CUDA 峰值分配基本相同：batch=1 为 1244.75 MiB、batch=7 为 2296.34 MiB，主要收益是速度。782 条 tegrastats 记录中系统 RAM 峰值为 7164 MB、swap 为 0、GPU 最高约 59.38°C、最高结温约 61.84°C；没有更改功耗模式、频率或依赖。

目标帧距观测约 2 秒，而候选模型条件仍为 1 秒，因此表中的 LPIPS 仅衡量目标匹配，不能当作一秒未来图像真值误差。DDIM 的目标分数接近 DDPM，但三组场景均改变最终候选；DPM-Solver++ 也改变其中两组。不能据此认定导航质量保持不变，默认仍为 DDPM。本次没有连接 ROS/DDS、发布运动指令或执行闭环导航。

机载原始结果、126 张七候选预测、计时日志、遥测和可复现的 `benchmark.py` 保存在 `/home/unitree/ndj/visualizations/opennwm/sampler-comparison-20261019/`。更新前仓库备份为 `/home/unitree/ndj/backups/OpenNWM-deployment-before-samplers-20261019.tar.gz`。机载系统日期仍为 1970 年，本次未更改；持续时间使用单调时钟，不受系统日期错误影响。Hydra 数据集 infer/plan 和实际导航入口尚未执行完整测试。

## 环境约束

不要在此环境直接安装根目录所述桌面版 `nwm/requirements.txt`，其中的 Torch/CUDA/Python 版本与这台 Jetson 不匹配。
使用 [NVIDIA JetPack 5 官方 PyTorch](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 的 `torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl`。
TorchVision 从 PyTorch 官方 `vision` 的 `v0.16.0` 源码构建，使用本机 CUDA 11.4，`TORCH_CUDA_ARCH_LIST=8.7`、`MAX_JOBS=2`、`BUILD_VERSION=0.16.0`、`PYTORCH_VERSION=2.1.0a0+41361538.nv23.06`，关闭未用的 FFmpeg/video-codec 扩展。
其余推理依赖见 `jetson-requirements.txt`。该环境用于图像预测和 LPIPS 评分，不包含训练、DreamSim/FID 完整评测或机器人控制环境。
使用本页的静态图像入口；原有 `run.py nwm infer/plan` 数据集评测入口仍按仓库原要求使用 Python 3.10/3.11。推理模块的 `EvalDataset` 改为按需导入，避免单图推理加载 Python 3.10 的训练数据组件。

脚本强制 Hugging Face 离线模式，从已下载的本地路径加载全部权重；`TORCH_HOME` 自动指向 LPIPS 所需的 AlexNet 缓存。清单 `models/opennwm-assets.json` 保留模型来源和 SHA256。
Stage 2 权重中的训练 RNG 状态引用 NumPy 2 的模块名；入口在读取可信的本地 checkpoint 时将该模块名映射到 NumPy 1.24 的等价实现，不改动权重文件，也不恢复训练状态。模型 EMA 参数使用 `strict=True` 加载。

## 当前照片与多轨迹视频

`jetson_video.py` 从运行目录内的 `source.png` 读取照片，同时生成 OpenNWM 和 NWM-B 的原地、直行、左弧线、右弧线预测：

```bash
python -B deploy/jetson_video.py \
  --directory /home/unitree/ndj/visualizations/opennwm/你的运行目录 \
  --steps 50 --batch-size 8
```

每个视频恰好包含 8 张预测图，4 FPS、2 秒，预测时刻为 0.25、0.5、…、2.0 秒；视频入口默认 50 步扩散采样。使用固定初始照片直接预测各时刻的累计相对位姿；初始图重复为四帧上下文，VAE 采用确定性 posterior mode，各轨迹、时刻和模型使用相同扩散噪声。此模式与仓库网页的 direct 模式一致，不把预测帧循环作为下一步输入。

直行的总距离为 2 个训练 waypoint-spacing 单位；左右弧线弧长同为 2，终点偏航为 ±45°。这不是经过本机标定的米制轨迹，更不是控制指令。精确位姿、模型来源和预测设置保存在 `manifest.json`。

生成 8 个独立 H.264 MP4、一个并排对照视频、轨迹图、模型输入图和清单；通过本机 GStreamer 编码，不保留中间帧和日志。原始照片保留供对照。机载可视化统一放在 `/home/unitree/ndj/visualizations/opennwm`；本机接收目录为当前任务的 `outputs/go2-visualizations`。读取 Go2 前视画面只接收 `230.1.1.1:1720` 的 H.264 RTP 视频流，不初始化运动客户端。

## 只选 waypoint，不执行运动

`jetson_select.py` 复用上述视频中的四条候选轨迹。它读取每个模型各轨迹的第 8 帧（预测时刻 2 秒），与真实目标视角照片计算 LPIPS/AlexNet 距离，分别选出 OpenNWM 和 NWM-B 中分数最低的候选。不导入机器人 SDK，不发布 ROS/DDS 消息，没有运动执行开关。

没有目标照片时，只准备可视化，不产生评分和选点结果：

```bash
python -B deploy/jetson_select.py \
  --run-directory /home/unitree/ndj/visualizations/opennwm/camera-20261017-111045 \
  --prepare-only \
  --output /home/unitree/ndj/visualizations/opennwm/waypoint-preparation
```

有目标照片后，在机载推理环境中执行（将目标路径替换为真实存在的照片）：

```bash
python -B deploy/jetson_select.py \
  --run-directory /home/unitree/ndj/visualizations/opennwm/camera-20261017-111045 \
  --goal /实际路径/目标视角照片.png \
  --output /home/unitree/ndj/visualizations/opennwm/waypoint-selection
```

每次仅保存 `selection-preview.png`、`waypoints.png`、`selection.json`。输出目录必须尚不存在。清单保留完整排名、分数、第一 waypoint、末端 waypoint 和全部 8 个 waypoint；位姿均相对于起始假设机体坐标系累计表示（x 前、y 左、偏航逆时针为正），不是逐步增量。`eligible_for_execution` 始终为 false。

这个入口是固定四候选、末端图像评分的最小选点实验，不是仓库原版 CEM 自回归规划器。原版规划会采样候选、多次随机预测并更新候选分布；本入口复用已保存的单次 direct 预测，不做目标 VAE 重建。视频压缩会轻微影响图像评分。最低 LPIPS 只表示该候选的预测终点在视觉上更接近目标，不代表无碰撞、可通行或能到达目标，也不作为模型间的成功率比较。

用于真实导航前仍需独立验证：相机到机体的外参、waypoint 到米的尺度和时间定义、里程计、深度/点云障碍与落差检测、轨迹跟踪、速度与加速度限制、传感器和指令超时停止、人工接管与紧急停止，以及推理延迟预算。模型选点应经过独立的安全过滤和局部控制器；不要把归一化 waypoint 直接传给运动接口，也不要把 4 FPS 的视频播放率当作规划实时频率。

## 已采集轨迹的离线起终点测试

`jetson_episode.py` 仅读取文件，没有机器人 SDK、ROS/DDS 或运动执行入口。先在本机准备 3 对真实帧，再将轨迹和准备目录传到机载推理。每对及其历史帧位于同一个连续有效片段，起终点间隔约 2 秒；按接收时间选最近帧，误差不超过 40 ms。距离使用局部里程计的平面欧氏位移，朝向差先环绕到 [-π, π]，同时要求 **0.25 m ≤ 距离 ≤ 0.5 m 且 15° ≤ 朝向差绝对值 ≤ 30°**，包含边界。依次选左转、右转和第三个窗口，优先接近区间中心 0.375 m / 22.5°，各开始时刻间隔超过 5 秒。这些代表性样本不构成随机性能基准；上一轮 OR 条件实验作为历史结果保留，新规则另建目录运行。

```bash
python3 -B deploy/jetson_episode.py prepare \
  --raw /原始数据/episode_名称 \
  --cleaned /清洗数据/episode_名称 \
  --output /新的可视化目录
```

准备目录包含 `pairs.json` 和 `pairs.png`；清单保存原始/清洗帧编号、时间戳、距离、角度和输入图像 SHA256。将整个原始 episode、清洗结果和准备目录复制到机载后，在机载推理环境运行：

```bash
/home/unitree/ndj/envs/opennwm/bin/python -B deploy/jetson_episode.py infer \
  --raw /机载原始数据/episode_名称 \
  --output /机载可视化目录 \
  --steps 50 --batch-size 7
```

输入为开始时刻之前约 -0.75、-0.5、-0.25、0 秒的四张真实 RGB，采用仓库的 4:3 中心裁剪、224×224 resize 和确定性 VAE 编码。目标帧只用于 LPIPS 评分，实际未来轨迹不作为模型候选输入。两个模型都同时接收图像历史、动作和时间条件：预测时长固定为 **1 秒**，按模型 4 FPS 的时间约定，`Δn=4`，归一化时间固定为 `k=rel_t=4/128=0.03125`，不随实际采集时间间隔变化。这个 k 与 DDPM 的 50 步采样不是同一个量。`pairs.json` 和 `results.json` 中的 `horizon_seconds` 仅记录原始起终点采集间隔；`model_horizon_seconds=1`、`model_frame_offset=4`、`normalized_k=0.03125` 记录模型实际使用的固定条件，8 个 waypoint 的时间为 0.125、0.25、…、1 秒。读取旧版准备清单时，推理同样强制使用这些固定条件，并在新的 `results.json` 中记录实际值；历史结果不自动改写。

每对使用同样的 45 个固定候选：1 个原地、4 档直行（0.5、1、1.5、2）、32 条左右弧线（4 档弧长 × 15°/22.5°/30°/45° × 左右）、8 个原地转向（上述 4 档角度 × 左右）。原来的 7 个候选名称及路径保留。推理和 LPIPS 均按 `--batch-size` 分批执行，避免增加候选数时一次性占用更多 GPU 内存。XY 仍是训练 waypoint-spacing 单位，不能当作米；yaw 为弧度。两个模型对每个候选分别预测终点图像，以浮点解码结果计算 LPIPS，不经过视频压缩。对同一对帧，候选和模型使用相同扩散噪声。它是有限候选选点实验，不是 CEM、闭环导航或避障验证。

候选弧线的弧长为 0.5、1、1.5、2 个训练单位，终点偏航绝对值为 15°、22.5°、30°、45°。其中原有弧长 2、转角 ±45° 的候选，起终点直线距离约 1.949 个训练单位，终点为 `[1.800633, ±0.745846, ±π/4]`。直行候选路径长度/净位移为 0.5、1、1.5 或 2；原地和原地转向候选平移为 0。对真实帧使用的米制筛选范围，不会自动转换候选动作尺度；当前没有声明训练单位等于 Go2 的米。

生成器实际收到的是各候选的末端相对位姿 `[x, y, yaw]`，不是整条 8 点路径。候选转角增加了 ±15°、±22.5°、±30°，以更细致地覆盖本次目标朝向变化；仍然是离散候选。

`results.json` 保存完整排名、选中 waypoint、8 个累计相对位姿、权重/脚本标识及耗时；6 张模型对照图按候选数量自动扩展网格，并标出被选候选。8 个位姿描述预设候选路径，只有终点图像参与预测和评分，不表示已经验证整条路径。脚本将结果打印到终端，无日志文件，始终标记 `eligible_for_execution=false`。已有结果目录拒绝再次推理覆盖。

## 参考视频导航：单入口、逐次确认

`jetson_navigate.py` 串联参考帧提取、实时相机和里程计、模型选点、在线尺度拟合、避障碎步定时执行、实时监看与记录。默认只在明确选择 `navigate` 模式后创建控制发布器；每个动作还必须在真实交互终端输入对应的 `MOVE 编号`。无自动确认参数，不接受管道输入，不自动站立。可切换 `--model opennwm`（默认）或 `--model nwm-b`。

机载所需环境已配置：`/home/unitree/ndj/envs/go2-ros` 是由官方 `unitree_api`、`unitree_go` 消息源码构建的 Foxy overlay；`/home/unitree/ndj/code/go2-runtime` 复用现有 `motion_control` 和 `go2_capture` 模块。脚本会自动加载 ROS 环境，不需要分别启动控制器、录像或推理程序。部署到另一台设备时，这些依赖也需要安装；入口不是脱离仓库和权重的独立程序。

默认参考数据是已录制的 `episode_20261017T050607_8c26ec7b`，读取其视频对应的原始帧与 `aligned_pose.csv`，取从首帧起的前 30 秒。按时间顺序，相对于上一个目标帧，满足 **(0.25 m ≤ 位移 ≤ 0.5 m 或 15° ≤ 转角绝对值 ≤ 30°)，并且位移 ≤ 0.5 m、转角绝对值 ≤ 30°** 时取下一目标。无效位姿、时间重置或超过 0.5 秒的数据断口拒绝使用。本数据得到 8 个目标，最后一个在 27.797 秒；余下约 2.173 秒未达到新的下限，记录在 `reference.json`，不擅自追加不符合阈值的终点。`--target-rule strict` 可显式选择原来的 AND 规则。

```bash
# 在机载终端运行：先生成参考帧，完全不连接控制接口
/home/unitree/ndj/envs/opennwm/bin/python -B /home/unitree/ndj/code/OpenNWM/deploy/jetson_navigate.py prepare \
  --output /home/unitree/ndj/visualizations/opennwm/navigation-reference-001

# 只读检查相机、里程计和遥控器，保存短视频；不会开启避障或发布速度
/home/unitree/ndj/envs/opennwm/bin/python -B /home/unitree/ndj/code/OpenNWM/deploy/jetson_navigate.py probe \
  --output /home/unitree/ndj/visualizations/opennwm/navigation-probe-001

# 只推理一个候选选择，持续录像，无运动发布器
/home/unitree/ndj/envs/opennwm/bin/python -B /home/unitree/ndj/code/OpenNWM/deploy/jetson_navigate.py preview \
  --output /home/unitree/ndj/visualizations/opennwm/navigation-preview-001

# 实机导航：机器人需事先由现场人员置于可行走状态，并对齐参考起点
/home/unitree/ndj/envs/opennwm/bin/python -B /home/unitree/ndj/code/OpenNWM/deploy/jetson_navigate.py navigate \
  --output /home/unitree/ndj/visualizations/opennwm/navigation-live-001
```

每次输出目录必须尚不存在。默认机载接口为 `eth10`，可用 `--interface` 指定实际接口。导航模型预测时间固定 0.5 秒（`Δn=2`，`k=rel_t=2/128=0.015625`），50 步 DDPM；实际运动时长由计划指令和固定速度计算，不要求机器人在 1 秒内走完。当前推理并非实时：9 候选约分钟级，45 候选约 7 分钟，推理时不保持上一动作的速度指令。

### 观测、候选与尺度

历史按动作完成事件积累，不再按时间均匀抽帧。开始保存初始帧 I0，每完成一次实际执行并停稳后，等待接收时间晚于动作结束的新相机帧，依次追加 I1、I2……。模型仍要求 4 槽位：依次为 `[I0,I0,I0,I0]`、`[I0,I0,I0,I1]`、`[I0,I0,I1,I2]`、`[I0,I1,I2,I3]`，以后只保留最新 4 张。少于 4 张时在左侧重复最早帧，清单记录真实数量、补齐数、动作提案编号及原始帧序号。不确认、确认过期、切换目标或选择静止不追加历史；若等待期间场景/位置变化超过原有门槛，旧提案作废并进入恢复提示，不再因该检查直接退出。输入 `replan` 明确从当前位置重建初始历史（当前新帧重复填充 4 槽位），原执行记录和历史文件保留；不会把等待帧追加为已执行动作。新提案仍须新的 MOVE 编号。输入 `q` 则结束。相机与位姿仍采用主机接收时钟，未声明曝光级硬同步。导航模型预测时间条件固定 0.5 秒；历史仍按动作事件积累，实际底盘执行时长仍由运动计划决定。

保守阶段使用 9 个非静止动作：直行 0.5、弧长 0.5 的左右 15°/22.5°、原地左右 15°/22.5°；只有开放完整 45 候选后才允许 stationary。XY 初始尺度为 **0.30 m/训练单位的保守执行假设**，yaw 初始尺度为 1。每个非零平移指令的净距离至少 **0.21 m**，每个非零转角至少 **15°**，纯直行不强加转弯，原地转向不强加平移。启动硬预算仍为 0.25 m/25°，后续为 0.5 m/50°；规划路径长度留 0.025 m、转角留 5° 的余量，实际路程距硬上限 0.015 m、实际转角距硬上限 4° 时停止。最小值是计划指令下限，定时执行及避障不能保证实际位移达到该值。

先对整个候选集应用物理幅度上下限，再反算有效模型动作并进行预测和评分；预测、执行计划、在线拟合使用同一个有效动作。平移与转向分别调整，使用 SE(2) 恒定机体速度模型重建有效路径，按弧长而非仅端点直线距离限幅。每轮 `candidate-proposals.json` 保存原始候选、有效路径、米制指令和速度时长，选中预测无需另行补算。

在线拟合采用过原点的线性模型：`[dx_m,dy_m] = s_xy × [x_model,y_model]`、`dyaw_rad = s_yaw × yaw_model`，用最多最近 60 个有效动作作正则化最小二乘；系数限制为 `s_xy∈[0.05,1]`、`s_yaw∈[0.5,1.5]`。每次执行均记录原候选、限位后的有效候选、下发米制目标、实际位姿差和拟合前误差。发生限位时用有效候选拟合；有效样本要求执行成功、非静止，且误差不超过 0.12 m/10°。失败、静止或明显离群的动作仅记录，不污染拟合；最终反馈缺失也记录失败原因。

开放全 45 候选要求：至少 12 个有效样本，至少 8 个含平移、8 个含转向，包含左右转；最近 8 次均有效，拟合前平移 RMSE ≤ 0.08 m、转角 RMSE ≤ 5°，单次误差 ≤ 0.10 m/8°，最近 8 次两项系数变幅均 ≤ 当前系数的 10%。条件不再满足时回退到小动作集合。**这只衡量已试动作的执行尺度一致性；不能从自己下发的命令与反馈中独立识别模型训练单位的真实米制含义，也不能保证越走越准确。** 当前参考路径以左转为主，可能始终无法取得左右覆盖，脚本会继续使用小动作。

### 确认、到达与停止

启动提示接受 `START` 或 `start`，输错会重新提示，`q` 退出；每次动作的 `MOVE 编号` 仍须完整匹配。终端每次打印候选、米制目标、硬限位、当前目标 LPIPS 和 `step-xxx/decision.png` 路径。输入完全匹配的 `MOVE 001` 等编号才执行；空输入、`yes` 或错误编号不执行。`arrived` 表示现场人员确认已经到当前参考目标并进入下一目标；`q` 退出。尚未标定可靠的视觉到达阈值，因此不把 LPIPS 最低或选中 stationary 自动当作到达。每次确认最多有效 120 秒；超时、机器人较推理起点移动超过 3 cm/5°，或当前图相对推理输入的 LPIPS > 0.15 时，作废提案并提示 `replan` / `q`。位姿超限会打印实测差值和阈值，`proposal-rejection.json` 保存前后位姿、时间戳和恢复选择；其它传感器、遥控接管及执行故障仍停止。

每次动作成功结束、停止并记录执行结果后，**先询问是否到达，再进行下一次推理或额外图像比较**：`arrived` 确认当前目标并切换下一目标（最后一个目标则结束）；回车或 `continue` 表示未到达，继续规划；`q` 退出。输错只重新询问，不启动推理。即使达到提案次数上限，也会先询问本次动作是否到达。选择 `arrived` 或 `q` 时跳过本次可选的 `image-comparison.json` 计算。

每次终端 MOVE 确认后，才调用宇树 `StaticWalk`（Sport API 1061）选择碎步模式，等待 0.5 秒并复查状态，然后开启并确认机载避障、获取 API 速度控制权。平移、转向均以固定机体速度通过避障接口执行；线速度不超过 0.12 m/s、角速度不超过 0.35 rad/s，持续时间按计划距离/速度计算（非零动作至少 1 秒），每 0.05 秒更新。完成条件是命令时长到期，**不再用位置/角度误差追踪或判断 waypoint 到达**；`execution.json` 明确记录 `waypoint_verified=false`。结束后停止并释放控制权，等待停稳仅用于安全收尾和获取实际位移，不校验是否到达指令终点。正常完成后仍可因实测偏差过大而不纳入尺度拟合。

里程计仍用于记录真实 dx/dy/dyaw、在线统计与累计路程/转角硬预算；相机、姿态安全、遥控器接管、自带避障状态和命令看门狗继续检查。控制循环延迟超过 0.2 秒会停止，不补发或延长动作；没有自动重试运动。API 的碎步模式请求不构成足端确已迈步的独立验证；本次修改仅做离线与只读检查。

状态过期、异常姿态/运控状态、遥控器接管、相机断流、录像队列溢出、磁盘不足、定位跳变或预算用尽均阻止/中止动作。运动期间可按 Ctrl+C 或使用实体遥控器接管。自带避障不能保证识别玻璃、细线、台阶或盲区；主机断电或网络断开时也无法保证停止请求送达。因此不能保证绝不碰撞，首次测试应在现场看护的平整、开阔区域进行。

### 输出与检查

- `navigation.mkv`：整个运行阶段的视频，含推理等待期；保留真实接收时间的 PTS，正常退出会完成封装。
- `video-timestamps.csv`、`trajectory.csv`：逐帧时间/RTP 信息和约 20 Hz 位姿记录。
- `navigation.log`：与终端同步的进度、推理日志、警告与错误。
- `reference.json/png`：参考目标序列、提取依据与未取样尾段。
- `step-xxx/`：历史帧、目标帧、原始选中预测、限位后预测、确认后执行记录、实际动作后图像与比较分数。
- `calibration.json`：全部动作观测、过滤原因、当前系数及是否允许全候选。
- `run-status.json`：运行状态与参数。失败不标记完成；文件只说明里程计反馈，不是外部真实精度证明。

```bash
# 无 ROS、GPU 或运动接口的检查
python3 -B -m unittest discover -s deploy/tests -v
```

控制接口依据：[宇树官方 ObstaclesAvoidClient](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/obstacles_avoid/obstacles_avoid_client.py)。现场实机导航和尺度准确性仍需在逐次确认下验收。

相机按原始主机接收时间选取真实历史帧。偶发重复或短暂倒序帧会丢弃并在 `navigation.log` 记录 sequence、RTP、SSRC、原始时间及回退量，不修改时间戳，也不纳入录像或模型历史。连续 5 帧异常或回退超过 0.5 秒仍锁存故障；超过 0.5 秒没有有效新帧仍阻止执行。正常退出会报告丢帧总数，收尾异常不会覆盖先前的原始错误。

### 浏览器实时监看

在当前电脑连接时使用 `ssh -L 8765:127.0.0.1:8765 unitree@192.168.123.18`，机载终端运行原导航命令后，在当前电脑浏览器打开 `http://127.0.0.1:8765`。机载电脑有桌面时也可直接访问该地址。脚本自动启动只监听 localhost 的只读页面，默认端口 8765；冲突时用 `--display-port 8766` 并相应调整 SSH 转发，`--display-port 0` 可关闭显示。

摄像头画面约每 150 ms 请求一帧，模型推理和终端等待输入期间继续刷新；相机过期或连接断开会明确标记。页面显示当前目标编号、参考视频帧号和时间，完整展示本轮实际输入的 4 个历史槽位及对应动作编号（不足 4 张时明确标注左侧重复补齐），并显示选中预测画面、候选数、动作幅度、限位与执行时长。历史帧固定到本次推理；刷新真实摄像头不会替换模型已使用的历史帧。切换目标时清除上一轮历史和预测，准备下一次真实输入。网页没有运动控制接口，所有 MOVE/arrived/q 仍在终端输入。

遥控数据：严格保留 40 字节、5551 包头、有限摇杆数值校验。等待/推理阶段的无效包仅丢弃，不刷新有效数据时间；必须恢复连续至少 0.2 秒的有效数据，且最新有效包不超过 0.5 秒，才能继续。停稳等待最多给 2 秒恢复窗口，持续缺失仍终止。MOVE 确认后、切换步态之前即进入执行保护：此后任何无效包都锁存故障，正常包不能自行解锁；实际遥控接管在任何阶段都锁存。日志记录无效包长度、包头、是否全零、累计次数及恢复情况，不放宽解码格式。

### 显示、预处理与推理耗时

相机模型视图、目标、历史、预测均展示为 224×224，顶部三块使用同样的显示尺寸。原始相机/目标/历史统一经过与训练代码相同的 4:3 中心裁剪（含必要的补黑边）和 bilinear resize，再用于显示或张量归一化；预测已经是 224×224，不再重复裁剪。每轮新增保存 `model-input-history-0..3.png`、`model-input-goal.png`，动作后保存 `model-input-after.png` 和 `post-action-history.json`，便于核对真正输入。原始录像及原始帧仍保留。

历史 VAE 潜变量按预处理像素缓存，初始重复 4 槽只编码一次；以后一般只编码新动作帧，缓存最多 8 张。`selection.json` 和终端记录新增编码数量与耗时。默认仍为 50 步，可用 `--steps 20` 或 `--steps 10` 评估减少采样步数的速度/质量取舍；它会改变生成图和候选排序，不应把单例加速结果视为导航质量验收。当前已经使用 bfloat16 autocast 和 inference_mode。

标定筛选门槛保持不变：执行成功、有效非静止动作、平移误差 ≤ 0.12 m、转角误差 ≤ 10°。`exclusion_reasons` 和终端明确区分执行失败、平移超差、角度超差及静止/过小；命令按时执行结束不等于实际位移与指令一致。该统计继续用于开放 45 个候选，独立执行补偿不受这个误差筛选门槛限制。

导航默认启用独立执行补偿，无需新增启动参数。模型到米/度的尺度在本次运行固定为初始值（平移 0.30，转向 1.0），防止模型尺度学习和底盘补偿互相抵消。平移与转向各有独立补偿系数，每次运行从 1.0 开始，不加载旧记录；正常结束的定时动作根据实际位移更新系数，采用 25% 新估计的平滑、单次相对变化不超过 20%、总倍率 0.5–2.0。直线动作主要延长时长；组合动作按两个系数调整指令积分量，再共同计算时长和速度。速度始终不超过 0.12 m/s、0.35 rad/s，动作时长不超过 15 秒，物理路程/转角限位和 MOVE 确认保持有效。生成模型的动作条件仍是补偿前的预期目标，不能把预测图当作真实执行结果。

执行失败、反馈无效、已知避障干预不学习；单轴实际进展不足 0.02 m / 2°、反向运动不增加该轴补偿，明显横向或转向偏差时不更新平移系数。当前接口不能可靠识别所有自动避障减速，未知避障影响仍可能混入样本，因此倍率和单次调整有上限；不会自动重试或绕过避障。到时结束，不使用 waypoint 到达闭环。达到补偿上限仍欠执行时，不继续增加倍率。

终端在确认前打印两个补偿倍率、原始/补偿后时长和速度，执行后打印学习原因。运行目录的 `execution-compensation.json` 保存所有补偿观测；每步 `execution-compensation-update.json` 保存更新前后值。`selection.json` 和 `execution.json` 保存该次实际采用的计划与倍率。

导航停稳检查修复：运动中仍在硬限位前预留 0.015 m / 4° 提前停车；定时动作正常结束并停稳后，最终检查使用真实硬限位（保守阶段 0.25 m / 25°），不再将进入停车预留区本身误判为执行失败。累计路程、累计转角与净位移均受检查；运动中已触发的故障不会被最终检查清除。失败日志输出 error、budget_error、feedback_error。仅导航入口改为 0.5 秒时间条件，离线 episode 实验保留原设置。

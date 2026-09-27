# OpenNWM

LAM 与多阶段 NWM 的训练、评测和断点续跑。保留原模型、损失、动作语义和权重格式；数据与权重放在仓库外，运行不依赖旧 DreamDojo / CompACT 仓库。

| 要做的事 | 使用说明 |
| --- | --- |
| LAM：Action / Pixel / DINO / Pixel+Action | [LAM](lam/README.md) |
| NWM：直接训练、Stage 1、Stage 2、预测与规划 | [NWM](nwm/README.md) |
| 下载数据、权重，复用已有缓存 | [数据与权重](DATA.md) |
| 查看与原实现的对照结果 | [测试记录](tests/VALIDATION.md) |
| 浏览论文、预测视频和数据集交互演示 | [项目网页](website/README.md) |

## 1. 安装

两个环境分别安装，所有命令从仓库根目录执行：

```bash
git clone git@github.com:B0B8K1ng/OpenNWM.git
cd OpenNWM
conda create -n open-lam python=3.10 -y
conda create -n open-nwm python=3.10 -y
conda run -n open-lam python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n open-lam python -m pip install -r lam/requirements.txt
conda run -n open-nwm python -m pip install -r nwm/requirements.txt
```

## 2. 设置路径

已有数据直接复用；没有数据先按 [DATA.md](DATA.md) 下载。

```bash
python setup_paths.py \
  --data-root /data/navigation --work-root /data/opennwm \
  --lam-python /path/to/envs/open-lam/bin/python \
  --nwm-python /path/to/envs/open-nwm/bin/python
source .env
```

只需修改这四个路径。`.env` 不提交 Git；每个人可使用自己的磁盘和环境。生成器同时解压仓库附带的 NavAnywhere 固定 train/val recipe，并从有标签数据的原 train 清单留出 5% 作为 val，原 test 不变。已有独立验证集时传 `--split-root /data/splits`；复现旧实验时须使用同一 split，重新留出 val 会改变训练配方。15 源 NavAnywhere 使用 `--nav-profile 15`。

之后每个新终端执行 `source .env`。已有 VAE/DINO 用 `--vae`、`--dino-repo`、`--dino-weights` 指定；缓存路径可直接编辑 `.env`。示例见 [.env.example](.env.example)。

## 3. 训练、评测、恢复

```bash
# LAM：--variant 1/2/3/4；DINO 先准备教师权重。
python run.py lam train --variant 2 --run-dir "$OPENNWM_WORK_ROOT/runs/lam-pixel"
python run.py lam train --variant 2 --run-dir "$OPENNWM_WORK_ROOT/runs/lam-pixel" --resume
python run.py lam eval all \
  --checkpoint "$OPENNWM_WORK_ROOT/runs/lam-pixel/checkpoints/best.ckpt" \
  --output-dir "$OPENNWM_WORK_ROOT/eval/lam-pixel"

# NWM 直接训练；Stage 1/2 所需缓存和模式见 NWM 使用说明。
python run.py nwm train --mode baseline --gpus 1 --run-dir "$OPENNWM_WORK_ROOT/runs/baseline"
python run.py nwm train --resume "$OPENNWM_WORK_ROOT/runs/baseline" --gpus 1
# 原始 NWM 配方使用 --mode original（默认 CDiT-XL）。
```

所有训练定期验证并输出指标和效果图，结束时补做一次验证。`latest` 用于续训，`best` 按验证指标选择，默认只保留最近 3 个定期权重；NWM 另保留阶段切换权重及各阶段最佳权重。训练恢复模型、优化器、调度器、随机状态和数据进度；评测重跑同一命令接着已完成的部分。精确续训须保持数据、batch、GPU 数和配置一致。

## 来源与复现范围

LAM 来自 DreamDojo `8f3eeb7` 的 `external/lam_project`；多阶段 NWM 来自 CompACT `8367a5e`；`--mode original` 保留本机 Meta NWM `3f6cd8e` 的训练配方、四数据集顺序和采样行为，另有数值对照。上游许可见 [licenses/](licenses/)。日常运行不需要旧仓库或额外检查步骤。

迁移测试覆盖计算、梯度、优化器更新、阶段切换和恢复；真实入口另做小规模训练与评测。完整训练的收敛指标仍须通过相同数据、环境和种子的完整实验确认，具体环境、结果与覆盖范围见 [测试记录](tests/VALIDATION.md)。

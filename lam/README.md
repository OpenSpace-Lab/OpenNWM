# LAM

保留四种原始 LAM 模型、损失、数据采样和正式评测指标。所有命令在仓库根目录执行。

## 安装与路径

先按[根目录说明](../README.md)安装环境并运行 `setup_paths.py`，然后在每个新终端加载路径：

```bash
source .env
```

下方命令直接复用 `.env` 中的数据、权重和工作目录。每个命令均支持 `--help`，路径也可通过 `--data-root`、`--split-root` 等参数覆盖。

## 数据

有动作的数据沿用原始格式：

```text
$NWM_DATA_ROOT/
├── recon/<trajectory>/{0.jpg, 1.jpg, ..., traj_data.pkl}
├── sacson/<trajectory>/{0.jpg, 1.jpg, ..., traj_data.pkl}
├── scand/<trajectory>/{0.jpg, 1.jpg, ..., traj_data.pkl}
└── NavAnywhere/<dataset>/<trajectory>/{0.jpg, 1.jpg, ...}

$NWM_SPLIT_ROOT/<recon|sacson|scand>/<train|val|test>/traj_names.txt
```

`traj_data.pkl` 包含逐帧 `position` 和 `yaw`；清单每行一个轨迹相对目录名。NavAnywhere 为连续编号图片，无须动作标签；独立挂载可设置 `NWM_NAVANYWHERE_ROOT` 或传 `--navanywhere-root`。帧中心裁剪为 4:3 后缩放至 240×320。

LAM 与 NWM 共用 `$NWM_SPLIT_ROOT` 下的清单，可由根目录 `setup_paths.py` 生成，也可指定已有 split。训练只读取 `train`，周期验证读取独立的 `val`，训练本身不会重新划分，也不会用 `test` 选最佳模型。复现旧实验时须使用同一份 split。也可用 `--eval-config /data/val.yaml` 指定验证源：

```yaml
source_specs:
  - data_root: /data/navigation/recon
    split_manifest: /data/splits/recon/val/traj_names.txt
    display_name: RECON
    waypoint_spacing: 0.25
```

## 训练与续跑

| variant | 模型 | 默认数据配方 |
| --- | --- | --- |
| 1 | ActionOnlyLAM | `legacy`：RECON、HuRoN、SCAND |
| 2 | PixelLAM | `navanywhere`：13 个 NavAnywhere 源 + 3 个动作源 |
| 3 | DINOFeatureLAM | 同上；在线 DINOv3 特征 |
| 4 | PixelActionLAM | 同上；原校准动作损失权重 |

```bash
python run.py lam train --variant 2 --run-dir "$OPENNWM_WORK_ROOT/runs/lam-pixel" --devices 1
# 其他模型只需更换 --variant；13 源扩展为 15 源使用 --profile navanywhere15。
# 原三动作源配方：--profile legacy。

# 原命令加 --resume，恢复模型、优化器、调度器和训练进度。
python run.py lam train --variant 2 --run-dir "$OPENNWM_WORK_ROOT/runs/lam-pixel" --devices 1 --resume
# 指定旧项目或其他完整 Lightning checkpoint：--resume /weights/model.ckpt
```

运行目录保存解析后的 `config.yaml`、固定 `validation_pairs.json`、TensorBoard 日志与重建图。
默认每 1,000 步验证并保存 `checkpoints/latest.ckpt`，验证损失改善时保存 `best.ckpt`；每 10,000 步保存 `step=*.ckpt`，仅保留最近 3 个。所有 checkpoint 均包含完整训练状态。
可用 `--eval-every`、`--save-every`、`--periodic-every`、`--keep-periodic` 调整；`--keep-periodic 0` 保留全部定期 checkpoint。验证报告包含对应模型的损失、PSNR/SSIM、动作 MAE 或特征相似度。结束时补做验证与保存。

`--batch-size` 是每 GPU batch；`--max-steps`、`--lr`、`--precision`、`--grad-accum` 可覆盖默认值。`--config override.yaml` 可覆盖模型规模和数据参数。`--dry-run` 只写配置。多机可传 `--nodes`、`--node-rank`、`--master-addr`。可选 `--wandb project_name`，默认使用本地 TensorBoard。

## DINOv3 权重

variant 3 需要固定的 DINOv3 ViT-L/16 教师。路径已在 `.env` 配置；已有仓库时跳过 clone，已有权重会直接复用。其他 LAM 不需要它。

```bash
mkdir -p "$(dirname "$DINO_REPO_PATH")"
git clone https://github.com/facebookresearch/dinov3.git "$DINO_REPO_PATH"
git -C "$DINO_REPO_PATH" checkout 346f38fee679c56a6888f91c51670fae61d364e0
python run.py lam prepare-dino --output "$DINO_WEIGHTS_PATH"
python run.py lam train --variant 3 --run-dir "$OPENNWM_WORK_ROOT/runs/lam-dino"
```

下载支持断点续传。`legacy` DINO 配方保留原预计算特征训练方式：先在上述训练命令中加 `--profile legacy --dry-run` 生成配置，再执行：

```bash
python run.py lam precompute-dino \
  --base-config "$OPENNWM_WORK_ROOT/runs/lam-dino/config.yaml" \
  --cache-root "$OPENNWM_WORK_ROOT/cache/dino"
```

实际训练加 `--profile legacy --dino-cache "$OPENNWM_WORK_ROOT/cache/dino"`。周期验证和正式评测仍需要教师仓库与权重，按原变换从验证图像提取特征。

## 正式评测

```bash
python run.py lam eval all \
  --checkpoint "$OPENNWM_WORK_ROOT/runs/lam-pixel/checkpoints/best.ckpt" \
  --output-dir "$OPENNWM_WORK_ROOT/eval/lam-pixel" \
  --data-root "$NWM_DATA_ROOT" --split-root "$NWM_SPLIT_ROOT"
```

默认沿用原测试协议 ID、ID-CD、OOD、OOD-CD：训练域为 RECON/HuRoN/SCAND，OOD 为 TartanDrive（数据目录 `tartan`，清单目录 `tartan_drive`）。可用 `--config lam/config/navigation_eval_go_stanford.yaml` 切换 GO Stanford。动作、像素、特征分别采用原指标、控制组与 bootstrap 汇总，输出 `summary.md/json/csv` 和逐样本结果；`--num-visual 8` 增加重建可视化。

自定义评测 YAML 可直接填写各数据源和运行参数；命令行参数优先，数据路径环境变量次之，其余保留 YAML 设置。

再次执行同一评测命令，按已保存 batch 继续；输入 checkpoint 或配置变更时请换输出目录。可用 `prepare-manifest` 先固定采样，再给不同模型使用相同的 `--manifest`。LPIPS 首次运行会下载权重到 `.env` 配置的 `LAM_EVAL_ASSET_ROOT`；离线使用前可先运行 `python run.py lam eval prepare-assets --asset-root "$LAM_EVAL_ASSET_ROOT"`。`--disable-lpips` 仅用于快速调试，此时结果不含完整原评测指标。

NWM 的 LAM 特征衔接和各阶段训练见 [NWM 使用说明](../nwm/README.md)。

开发测试可选安装 `"$LAM_PYTHON" -m pip install pytest==9.1.1`，再执行 `"$LAM_PYTHON" -m pytest lam/tests tests/test_lam_resume.py -q`。

# NWM

保留 CompACT 的 CDiT、Gaussian diffusion、SD-VAE、数据采样、动作编码、损失、EMA 与评测算法。将机器专属启动脚本收敛为一个入口；新增验证集监控、完整断点、权重保留和本地曲线。旧工程的模型和损失没有重新实现。

## 环境与路径

推荐 Python 3.10/3.11，先按设备安装 PyTorch 2.6.0 + torchvision 0.21.0，再执行：

```bash
python -m pip install -r nwm/requirements.txt
export NWM_PYTHON=/absolute/path/to/env/bin/python
export NWM_DATA_ROOT=/absolute/path/to/navigation_data
export NWM_SPLIT_ROOT=/absolute/path/to/train_val_splits
export NWM_VAE_PATH=/absolute/path/to/sd-vae-ft-ema  # 也可用 stabilityai/sd-vae-ft-ema
export NWM_OUTPUT_ROOT=/absolute/path/to/runs
```

路径均可由环境变量或 Hydra `key=value` 设置。建议使用绝对路径。数据/大权重放仓库外；原始 split 索引在 `data_splits/`。推荐依赖使用公开稳定版本；原机 NWM 为 PyTorch 2.15 nightly，依赖差异不代表跨版本 GPU 逐位一致，验证记录见根目录。

NWM 数据格式：`$NWM_DATA_ROOT/{recon,sacson,scand,tartan}/轨迹/{0.jpg,1.jpg,...,traj_data.pkl}`。`sacson` 对应 HuRoN，`tartan_drive` 的数据文件夹名为 `tartan`。split 为 `$NWM_SPLIT_ROOT/数据集/{train,val,test}/traj_names.txt`。训练配置的 `test` 字段历史上表示周期评测集，本库默认指向 **val**；正式推理/评测独立使用 **test**。可逐项覆盖 `dataset.datasets.recon.train=...` / `.test=...`。已有独立 val 时直接使用；新从 train 留出 val 会改变数据配方，根目录工具可固定种子生成并记录来源。

## 训练

以下命令均从仓库根目录执行，末尾可添加 Hydra 参数。`--gpus` 为当前任务使用的卡数，配合 `CUDA_VISIBLE_DEVICES`。

```bash
python run.py nwm train --mode baseline --gpus 1 --run-dir /runs/baseline
python run.py nwm train --mode original --gpus 8 --run-dir /runs/nwm-original
python run.py nwm train --resume /runs/baseline --gpus 1
```

`baseline` 为 CompACT CDiT-B 配方；`original` 使用 CDiT-XL、原 NWM 优化器配方并关闭多类型动作适配器。原本地 NWM 与 CompACT 有模型、数据代码差异，`original` 表示原配方的兼容入口，不能将两个仓库声称为逐行相同的实现。

| 模式 | 含义 |
|---|---|
| `timept` | Stage 1，无动作预训练 |
| `geopt` / `idmpt` | Stage 1，几何 / IDM 离线代理动作 |
| `latentpt` / `latentonlypt` | Stage 1，LAM 动作（混合无动作 / 仅动作） |
| `time_reset` / `geo_reset` / `idm_reset` / `latent_reset` | Stage 2，新 real adapter warmup → 联合微调 |
| `latent_align` | Stage 2，LAM teacher embedding alignment → 联合微调 |
| `latent_real_to_latent` | Stage 2，real→latent 映射 → 联合微调 |
| `latent_state_controller` | Stage 2，状态条件 controller → 联合微调 |
| `no_pretrain` | 无 Stage 1 的随机初始化对照 |
| `go2_adapt` | 已有完整 NWM 权重适配 Go2 |

Stage 1 需要 NavAnywhere 帧目录、**不重叠的** train/val recipe、训练 SD-VAE posterior cache；有动作模式还需要对应 train/val proxy cache：

```bash
export NWM_NAVANYWHERE_ROOT=/data/navanywhere
export NWM_NAVANYWHERE_RECIPE=/data/recipes/train.json
export NWM_NAVANYWHERE_VAL_RECIPE=/data/recipes/val.json
export NWM_NAVANYWHERE_VAE_LATENT_ROOT=/data/cache/nav-vae
export NWM_NAVANYWHERE_VAL_PROXY_ROOT=/data/cache/nav-val-lam
python run.py nwm train --mode timept --run-dir /runs/timept seed=20260901
python run.py nwm train --mode latentpt --run-dir /runs/latentpt seed=20260901 proxy.storage_path=/data/cache/nav-train-lam
```

Stage 2 用四数据集 posterior cache，以及对应 Stage 1 checkpoint：

```bash
export NWM_FINETUNE_VAE_LATENT_ROOT=/data/cache/nwm-vae
export NWM_FINETUNE_LATENT_ROOT=/data/cache/nwm-lam
python run.py nwm train --mode latent_reset --run-dir /runs/reset finetune.stage1_checkpoint=/runs/latentpt/checkpoints/best.pth.tar
python run.py nwm train --mode latent_align --run-dir /runs/align finetune.stage1_checkpoint=/runs/latentpt/checkpoints/best.pth.tar
python run.py nwm train --mode no_pretrain --run-dir /runs/no-pretrain
```

所有训练周期及阶段结束时评估固定 val 样本与固定噪声，外围恢复 Python/NumPy/Torch RNG；Stage 1 默认已开启。默认 `eval_every=5000`、`training.eval_num_batches=4`，可设置 `training.eval_batch_size=...`。小验证集需减小这两个 batch 参数。预览位于 `viz/`（上下文/目标/生成图），指标为 `metrics.jsonl`，曲线为 `curves.png`；WandB 默认关闭。

`checkpoints/latest.pth.tar` 可续训，`best.pth.tar` 为验证指标最低权重，Stage 2 另存 `best_warmup` / `best_joint`，`transition.pth.tar` 保留阶段切换。周期权重默认保留 3 份，`training.keep_periodic=0` 保留全部。恢复包含优化器、scheduler、scaler、EMA、每卡 RNG 和数据游标；保持 GPU 数、batch、数据及配置一致才能精确接续。旧权重如不含完整状态只能按原逻辑恢复。

## 准备缓存

可直接复用旧工程缓存。以下命令生成最小兼容格式，每个完整轨迹原子保存，重跑跳过已完成轨迹。SD-VAE 固定 batch=128、BF16、posterior mean/logvar；训练时仍在线采样 posterior。

```bash
python run.py nwm recipe --root /data/navanywhere --output /data/recipes/all.json --seed 20260901
python run.py nwm cache-vae --data-root /data/navanywhere --recipe /data/recipes/train.json --output-root /data/cache/nav-vae --vae-model "$NWM_VAE_PATH"
python run.py nwm cache-vae --data-root "$NWM_DATA_ROOT" --split-root "$NWM_SPLIT_ROOT" --output-root /data/cache/nwm-vae --vae-model "$NWM_VAE_PATH"
# LAM 缓存工具需要 LAM 依赖，使用 LAM 环境运行 prepare.py：
"$LAM_PYTHON" nwm/prepare.py lam --data-root /data/navanywhere --recipe /data/recipes/train.json --output-root /data/cache/nav-train-lam --checkpoint /weights/lam.ckpt
"$LAM_PYTHON" nwm/prepare.py lam --data-root "$NWM_DATA_ROOT" --split-root "$NWM_SPLIT_ROOT" --output-root /data/cache/nwm-lam --checkpoint /weights/lam.ckpt
```

recipe 命令只生成一份 inventory；train/val 必须使用互斥轨迹的两个 recipe，已有原配方最便于复现。LAM 缓存保存 `z_mu` 原始均值、current→goal、绝对 offset≤8，格式 `{source}/{trajectory}.pt`，`frame_pairs:[N,2]` 与 `motion:[N,32]`。Geo/IDM 使用旧离线代理缓存相同接口，维度分别 3/7；其外部几何估计器不属于 NWM 模型本体。训练 offset 仍为 [-64,64]，局部 proxy mask 逻辑保持原样。

## 评测

推理支持 `time` 与 `rollout`；生成图像按样本恢复，已生成文件跳过且噪声由样本确定。训练目录需有 `.hydra/config.yaml` 和 checkpoint。示例一次生成 time 的 GT 与预测：

```bash
python run.py nwm infer exp_dir=/runs/latentpt ckp=best eval_type=time gt=1 output_dir=/data/eval
python run.py nwm infer exp_dir=/runs/latentpt ckp=best eval_type=time prediction_dir=/data/eval/pred
python run.py nwm eval gt_dir=/data/eval/gt exp_dir=/data/eval/pred 'datasets_to_eval=[recon,scand,sacson]' eval_types=time
python run.py nwm plan exp_dir=/runs/reset ckp=best
```

GT/预测实际目录由推理日志给出；指标工具输入目录下应为 `数据集/time/样本/*.png` 或 `数据集/rollout_1fps/样本/*.png`。LPIPS、DreamSim、FID 按原定义计算；每批持久化累积值、FID sufficient statistics 和游标，重跑继续，完成 JSON 直接复用。参数/输入改变请使用新输出目录或 `resume=false`。推理/规划记录权重的路径、大小、修改时间与关键配置，`latest`/`best` 改变时要求新输出目录，避免误用旧结果。规划保留逐样本结果以便恢复。首次 DreamSim/LPIPS/FID 会获取公开预训练权重；离线机器需提前准备相应缓存。

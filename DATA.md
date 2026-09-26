# 数据与权重

所有大文件放在 `--work-root` 或已有数据目录，Git 只保存代码、配置和固定采样清单。已经有原项目资源时，直接在 `.env` 填原路径即可。

## 数据

LAM 发布数据：[LittleBoss/LAM-Data](https://modelscope.cn/datasets/LittleBoss/LAM-Data)。安装 ModelScope CLI 后下载，断开后重跑相同命令即可继续：

```bash
python -m pip install modelscope
ms-hub download LittleBoss/LAM-Data --repo-type dataset \
  --local-dir /data/downloads/LAM-Data --max-workers 8
```

若仓库要求账户权限，先 `ms-hub login`。解压需要系统的 `zstd` 和 `tar`。NavAnywhere 每个包独立，动作数据按相同 archive 名拼接分片：

```bash
mkdir -p /data/navigation
cd /data/downloads/LAM-Data
find NavAnywhere -path '*/packs/*.tar.zst' -print0 \
  | xargs -0 -I{} tar --use-compress-program=unzstd -xf '{}' -C /data/navigation
for first in LAM-Data-{recon,scand,sacson}-*.tar.zst.part-00000; do
  archive=${first%.part-00000}
  cat "${archive}.part-"* | tar --use-compress-program=unzstd -xf - -C /data/navigation
done
```

解压结果：

```text
/data/navigation/
├── recon/<trajectory>/{0.jpg,1.jpg,...,traj_data.pkl}
├── sacson/<trajectory>/{0.jpg,1.jpg,...,traj_data.pkl}  # HuRoN
├── scand/<trajectory>/{0.jpg,1.jpg,...,traj_data.pkl}
├── tartan/<trajectory>/{0.jpg,1.jpg,...,traj_data.pkl}
└── NavAnywhere/<source>/<trajectory>/{0.jpg,1.jpg,...}
```

上述 LAM 发布包覆盖 RECON、HuRoN、SCAND 和 NavAnywhere；NWM 的四数据集训练及 LAM 默认 OOD 评测还需要 TartanDrive。TartanDrive、GO Stanford 等扩展数据按 [原 NWM 数据准备说明](https://github.com/facebookresearch/nwm) 使用其处理格式；本机这些数据已经准备好。split 的 `tartan_drive` 对应图片目录 `tartan`。

`setup_paths.py` 自动准备共享 train/val/test 清单与 NavAnywhere 13/15 源 recipe。recipe 固定轨迹、帧数、采样 seed；无需重新遍历图像生成。已有实验必须复用其原始 split 和 recipe。

## 权重

| 用途 | 是否需要 | 来源 |
| --- | --- | --- |
| SD-VAE | 所有 NWM 图像训练/预测 | [stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema) |
| DINOv3 ViT-L/16 | LAM variant 3 | `lam prepare-dino`，见 LAM README |
| 已训练 LAM | 直接评测或生成 NWM 代理动作 | 下方公开下载，或自己的 checkpoint |
| NWM Stage 1 | 从预训练结果开始 Stage 2 | 下方公开下载，或自己的 checkpoint |
| 原版 CDiT | 原 NWM 基线复现 | [facebook/nwm](https://huggingface.co/facebook/nwm) |

SD-VAE 可让首次运行自动下载，也可提前放到本地：

```bash
cd /path/to/OpenNWM
source .env
"$NWM_PYTHON" -c 'from huggingface_hub import snapshot_download; snapshot_download("stabilityai/sd-vae-ft-ema", local_dir="/data/models/sd-vae-ft-ema")'
# 在 .env 中设置 NWM_VAE_PATH=/data/models/sd-vae-ft-ema
```

公开的 LAM Action、Pixel+Action 100k 及 NWM LatentPT 200k 权重在 [LittleBoss/LAM-weights](https://modelscope.cn/models/LittleBoss/LAM-weights)。下载固定发布版本 `f18fc384`，已有文件复用，未完成文件继续下载：

```bash
python tools/download_weights.py --asset lam-pixel-action --asset nwm-stage1 \
  --output-root "$OPENNWM_WORK_ROOT/models"
# 可选：--asset lam-action 或 --asset dino
python run.py nwm import-checkpoint \
  --checkpoint "$OPENNWM_WORK_ROOT/models/nwm-latentpt-200000.pth.tar" \
  --run-dir "$OPENNWM_WORK_ROOT/runs/imported-stage1"
```

Pixel、DINO 两种 LAM 和最终 nav15 的权重本机均有；这里没有把未公开的 checkpoint 写成公开下载链接。其他机器可从已有存储复制，或使用对应配方训练。公开的 nav1 100k 权重与本机 nav15 60k 属于不同实验，不能混用其 LAM 缓存来声称同一实验复现。

## 缓存与本机现成资源

Stage 1 需要 VAE posterior 和相应代理动作缓存，Stage 2 需要四数据集缓存，生成命令见 [NWM README](nwm/README.md#准备缓存)。中断后重跑会跳过已完成轨迹。缓存必须与所选 LAM 权重和 recipe 对应；不要直接使用另一个权重生成的动作。

本机 `.env` 已连接现有数据、VAE、DINO、LAM 权重和 nav1 Pixel+Action 100k 对应缓存，资源链接位于 `$OPENNWM_WORK_ROOT`，没有复制大数据或修改旧实验。四种 LAM 权重可用 `$LAM_ACTION_CHECKPOINT`、`$LAM_PIXEL_CHECKPOINT`、`$LAM_DINO_CHECKPOINT`、`$LAM_PIXEL_ACTION_CHECKPOINT`；另有 `$LAM_NAV15_CHECKPOINT`。本机导入的 NWM 两阶段运行目录为 `$NWM_NAV15_STAGE1_RUN`、`$NWM_NAV15_STAGE2_RUN`，推理时使用 `ckp=latest`；这些是 nav15 实验，仅用于该实验的评测或接续。新机器只需运行路径设置命令并按需下载资源。

LPIPS、DreamSim、FID 的预训练网络首次评测会下载。离线运行时提前准备缓存：LAM 用 `lam eval prepare-assets`；NWM 的 Torch 缓存设 `TORCH_HOME`，DreamSim 缓存设 `NWM_MODEL_CACHE`。所有下载与训练输出均应放在容量足够的数据盘。

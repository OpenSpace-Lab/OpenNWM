# OpenNWM / NWM 的 DMD2 四步蒸馏

已完成的小规模训练、质量/速度实测及权重位置见 [实验结果](DMD2_RESULTS.md)。

ModelScope 下载：[OpenNWM-DMD](https://modelscope.cn/models/LittleBoss/OpenNWM-DMD)、
[NWM-DMD](https://modelscope.cn/models/LittleBoss/NWM-DMD)。每个发布包包含学生权重、
便携推理配置、四步采样器、使用示例及小样本测试结果，文件大小与 SHA256 已核验。

独立入口，不改原 `train.py`、DDPM 推理、网页服务、动作适配器或原权重。
方法参考 [DMD2](https://github.com/tianweiy/DMD2)，固定参考提交
`8d8fa55633d47cfb81bbc7a892e7248f9518763f`。这是面向 CDiT 的适配，
不是直接运行上游 SDXL 训练器，也不能将 SDXL 的蒸馏权重用于 NWM。

实现包含冻结真实评分教师、全参数学生、可训练 fake-score CDiT、
归一化 distribution matching 梯度、默认 10:1 的 critic/student 更新比、
真实图像与生成图像的 logistic GAN 损失，以及训练时模拟学生自身的前序采样。
学生每次训练均匀选择四个步骤之一，只对选中步骤反向传播；前序步骤停止梯度。
GAN 在 fake-score CDiT 的最后一层 token 上使用池化 MLP 分类头，替代上游 UNet 特征分类头。
NWM 不使用文本 CFG。训练使用确定性 VAE posterior mean，属于本实验的明确适配选择。

四步固定为 `[999, 749, 499, 249]`：预测 x0 后按下一时刻重新加独立噪声，
最后输出 x0。保留 CDiT 的 epsilon/variance 权重格式，但该采样器只读取 epsilon。
**不能**将蒸馏权重交给原 DDPM 入口并仅设置 `eval_diffusion_steps=4`；
须使用 `dmd2.FourStepSchedule.sample` 或下面的独立评测入口。

## 准备少量数据

从仓库根目录执行；先设置 `.env` 中的数据、VAE 和 Python 环境。
大文件写入 NAS。先用 `nvidia-smi` 检查空闲资源。

```bash
source .env
export DMD2_ROOT=/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019
CUDA_VISIBLE_DEVICES=0 "$NWM_PYTHON" -u -B nwm/distill_dmd2.py prepare \
  --source-run "$NWM_NAV15_STAGE2_RUN" --output "$DMD2_ROOT/data"
```

默认从 recon / sacson / scand 各取 128 个 train、8 个 val、8 个 test 轨迹，
每轨迹一个固定窗口；显式检查三个 split 无轨迹重叠。预测时间为未来 4/16 帧。
`manifest.json` 保存轨迹、帧号和源目录。`samples.pt` 保存固定 VAE 编码与原始 real action；
旧 NWM 的动作归一化在模型调用前应用，OpenNWM 由自己的 real adapter 处理。
小样本选择不等同于官方完整测试协议，也不代表所有时间范围。

## 两个模型分别训练

```bash
CUDA_VISIBLE_DEVICES=0 "$NWM_PYTHON" -u -B nwm/distill_dmd2.py train \
  --source-run "$NWM_NAV15_STAGE2_RUN" \
  --checkpoint "$NWM_NAV15_STAGE2_RUN/checkpoints/latest.pth.tar" \
  --data "$DMD2_ROOT/data/samples.pt" --output "$DMD2_ROOT/opennwm" \
  --generator-updates 200 --critic-ratio 10 --batch-size 2

export NWM_RELEASE_RUN=/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models/nwm-release
CUDA_VISIBLE_DEVICES=3 "$NWM_PYTHON" -u -B nwm/distill_dmd2.py train \
  --source-run "$NWM_RELEASE_RUN" \
  --checkpoint "$NWM_RELEASE_RUN/checkpoints/0100000.pth.tar" \
  --data "$DMD2_ROOT/data/samples.pt" --output "$DMD2_ROOT/nwm" \
  --generator-updates 200 --critic-ratio 10 --batch-size 2
```

这里 NWM 指官方发布的 CDiT-B 权重 `cdit_b_100000.pth.tar`。
训练量较小，结果应标记为 pilot，不能声称已经收敛或达到上游论文效果。
显存不足时可添加 `--activation-checkpointing`，或者降低 batch。
训练记录实际梯度、loss、更新时间与显存。非有限 loss/梯度会立即报错。

仅保留原子更新的 `latest.pt`（学生、critic、两个优化器及 RNG，可续训）
与训练完成的 `student.pt`（推理权重）。`--resume` 恢复同一目录，检查教师、数据哈希和关键参数；
增大 `--generator-updates` 可以继续训练。默认每 25 次学生更新保存一次。
原教师 checkpoint 从不覆写。不要删除 `latest.pt`，除非确定不需要续训。

## 质量与速度

```bash
CUDA_VISIBLE_DEVICES=0 "$NWM_PYTHON" -u -B nwm/benchmark_dmd2.py \
  --source-run "$NWM_NAV15_STAGE2_RUN" \
  --checkpoint "$NWM_NAV15_STAGE2_RUN/checkpoints/latest.pth.tar" \
  --student "$DMD2_ROOT/opennwm/student.pt" --data "$DMD2_ROOT/data/samples.pt" \
  --output "$DMD2_ROOT/opennwm-test" --label OpenNWM --repeats 3
```

NWM 同理替换 source-run/checkpoint/student/output/label。
四组共享输入、动作、样本种子和初始噪声：原教师 250 步 DDPM、原教师 4 步 DDPM、
原教师 4 步 DMD2 采样器、蒸馏学生 4 步 DMD2。后两组可单独衡量学习本身的收益。
每组先预热，样本内打乱方法执行顺序；默认重复 3 次，报告每样本中位数再求均值。
使用 batch=1、BF16 autocast 与 CUDA 同步计时，分别报告去噪耗时和在线 VAE 编码+去噪+解码耗时，
排除模型加载、磁盘读写和指标计算。

质量使用 PSNR、TorchMetrics Gaussian SSIM、AlexNet LPIPS，保存逐样本指标、
按数据集汇总及配对差值 bootstrap 区间。小样本不报告 FID。
`report.md`、`results.json`、`comparison.png` 和逐样本图像均放在指定输出目录。
这不是自回归 rollout 或导航成功率评测。

验证：`CUDA_VISIBLE_DEVICES='' "$NWM_PYTHON" -B -m pytest -q nwm/tests/test_dmd2.py`。

# DMD2 四步蒸馏小规模实测（2026-10-19）

已分别完成 OpenNWM Stage 2 与官方 NWM CDiT-B 的四步蒸馏，保存独立学生权重。两个训练和两个评测进程均退出 0。

**结论：四步推理显著加速，但本次短程训练尚未保持 250 步教师的感知质量。** PSNR/SSIM 提高与图像更平滑同时出现，LPIPS 变差；不能据此宣称整体质量提高或训练收敛。

## 实验条件

- 方法：[DMD2](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f) 的 CDiT 适配：真实/生成分布评分、10:1 critic/student 更新、条件 GAN、学生前序采样模拟。GAN 使用 CDiT token 池化分类头。
- 每模型 384 个训练窗口、200 次学生更新、2000 次 critic 更新，batch=2；四步时间点 `[999, 749, 499, 249]`。
- recon/sacson/scand 各 8 个 test 窗口，共 24 个；train/val/test 轨迹互斥，未来 4/16 帧直接预测。验证集保留，本次未用测试集调整参数。
- NVIDIA L20，batch=1，BF16 autocast；每方法先预热，三次计时取每样本中位数，再跨样本求均值。方法顺序随机交错，CUDA 同步。
- 固定输入、动作和初始噪声；VAE posterior mean；总耗时为在线 VAE 编码、去噪、解码，不含 I/O、模型加载和指标计算。
- 小样本不计算 FID；没有评测长时 rollout 或导航成功率。速度为本机当前配置实测。

## 蒸馏前后

| 模型 | PSNR ↑（前→后） | SSIM ↑（前→后） | LPIPS ↓（前→后） | 含 VAE 耗时 ms（前→后） | 加速 |
|---|---:|---:|---:|---:|---:|
| OpenNWM | 14.215 → 15.613 | 0.4515 → 0.5314 | 0.3598 → 0.4506 | 3574.22 → 84.05 | 42.53× |
| NWM CDiT-B | 13.951 → 15.437 | 0.4460 → 0.5237 | 0.4033 → 0.4938 | 3608.12 → 87.41 | 41.28× |

## 四步未蒸馏对照

| 模型 | 原教师 4 步 DDPM LPIPS ↓ | 原教师 4 步 DMD2 LPIPS ↓ | 学生 4 步 DMD2 LPIPS ↓ | 学生相对同采样器改善 |
|---|---:|---:|---:|---:|
| opennwm | 0.5008 | 0.4723 | 0.4506 | 4.58% |
| nwm | 0.5391 | 0.5095 | 0.4938 | 3.09% |

蒸馏对同样四步采样器的 LPIPS 有改善，但仍落后于 250 步教师。完整指标、按数据集统计、逐样本指标与配对差值 bootstrap 95% 区间见下列报告。

## 权重与复现

学生必须使用 `FourStepSchedule.sample`；仅在原 DDPM 入口设置步数为 4 不等价。原权重、原训练与服务入口未覆盖。

- opennwm：[学生权重](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/opennwm/student.pt)；[可续训断点](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/opennwm/latest.pt)；[完整报告](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/opennwm-test/report.md)；[机器可读结果](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/opennwm-test/results.json)；[对比图](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/opennwm-test/comparison.png)。

- nwm：[学生权重](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/nwm/student.pt)；[可续训断点](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/nwm/latest.pt)；[完整报告](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/nwm-test/report.md)；[机器可读结果](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/nwm-test/results.json)；[对比图](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/nwm-test/comparison.png)。

- [代码与复现命令](/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM/nwm/DMD2.md)；[执行记录、PID、参数、源文件哈希](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/execution.json)；[数据清单](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/data/manifest.json)。
- 新增 11 项测试通过；原有相关回归测试 23 项及 23 项 subtests 通过。语法检查与 `git diff --check` 通过。
- 临时参考源码已删除，无遗留 `.tmp` 文件；保留数据缓存、推理权重、续训断点、日志、指标和对比图，共约 11 GB，均存放 NAS。
- 已恢复此前暂停的 24 个 SpikeVLA 进程，恢复前核对 PID 启动时间：[恢复记录](/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019/resumed_spikevla.json)。

# OpenNWM

LAM 与分阶段 NWM 的训练、评测和续跑。由本机实际实验实现精简而来，保留原模型、损失、动作语义及 checkpoint 格式。

| 任务 | 入口与使用说明 |
| --- | --- |
| LAM：Action / Pixel / DINO / Pixel+Action | [lam/README.md](lam/README.md) |
| NWM：直接训练、Stage 1、Stage 2 | [nwm/README.md](nwm/README.md) |
| 原实现与新实现的数值对照 | [tools/](tools/) |

```text
OpenNWM/
├── run.py              # 统一入口
├── lam/                # LAM 模型、数据、训练、评测
├── nwm/                # NWM 模型、数据、训练、预测及规划评测
├── tests/              # 迁移验证
└── tools/              # 开发时旧新数值对照
```

## 使用

LAM 和 NWM 使用各自的 Python 环境，避免依赖版本相互影响。安装方式见各自 README。

```bash
export LAM_PYTHON=/path/to/lam-env/bin/python
export NWM_PYTHON=/path/to/nwm-env/bin/python
python run.py lam --help
python run.py nwm --help
```

数据、预训练权重与输出目录在命令或配置中指定；不需要原 DreamDojo / CompACT 仓库。
大数据、cache、checkpoint 放在仓库外。运行时不需要额外的扫描或校验命令。

## 训练与恢复

每种训练均使用固定验证集定期评测，保留最近、最佳及定期 checkpoint。
验证集用于监控与最佳权重选择，测试集仅用于最终评测。
训练恢复使用包含优化器与训练进度的完整 checkpoint；评测复用已完成的进度。
具体参数、保存位置与恢复命令见对应目录的 README。

## 来源

- 分阶段 NWM：CompACT `8367a5eb5bba9e8bdd2522a72b20557225f59db7`。
- LAM：DreamDojo `8f3eeb73d944a0504b1df5a753975ea130f3e3d8` 的 `external/lam_project`。
- 原始 NWM 配方与本机分阶段配方的参数不同，分别保留配置并注明。
- 上游许可见 [licenses/](licenses/)，继承的文件继续遵循其上游许可。

完整训练收敛后的指标是否一致，需要同一数据、环境和随机种子的完整实验；短测验证模型计算、训练更新、阶段切换和续跑行为。实际验证结果见后续测试记录。

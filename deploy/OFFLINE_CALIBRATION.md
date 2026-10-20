# 固定 15 候选：离线标定与导航

此版本导航不再在线学习动作尺度或执行补偿。`jetson_episode.py` 的历史 45 候选用于旧离线分析，实时导航使用 `calibrated_actions.py` 的独立固定 15 候选。

| 类型 | 名义路程（米） | 转角 | 个数 |
|---|---|---|---:|
| 静止 | 0 | 0 | 1 |
| 直行 | 0.25、0.50 | 0 | 2 |
| 左右圆弧 | 0.25、0.50 | 每档搭配 ±π/12、±π/4 | 8 |
| 原地左右转 | 0 | ±π/12、±π/4 | 4 |

圆弧距离是弧长，模型输入是起始机体坐标系下的终点。模型平移单位固定为 0.25 m/unit，角度是弧度。标定后实际输入 `[mean_dx/0.25, mean_dy/0.25, mean_dyaw]`，仍由各模型原有接口完成后续归一化。尺度 0.25 是选定的部署约定，不代表已验证训练数据与 Go2 的物理尺度一致。

## 标定程序会做什么

`plan` 只生成计划文件，不导入 ROS、不连接机器人、不加载模型。`collect` 使用已有 ROS、相机、里程计、遥控接管、独占运动锁和停止逻辑，机器人须已站立且可行走。

1. 为 14 个运动候选各执行一个试动作；若实测幅度/方向不合格，仅在此次离线标定中调整指令。每轮倍率变化限制在 0.8–1.2；每个候选默认最多 8 次试动作，无有效进展或反向时直接退出，不放大指令。
2. 冻结最终速度和持续时间，对该候选重复 3 次。静止候选观察 3 次，不发送运动指令。默认最少 59 次观察：14 次运动试动作 + 42 次运动重复 + 3 次静止；最多 157 次观察。每一次均需要终端中相应的 `MOVE NNN`，不会一口气自动执行整套动作。等待确认时可用遥控器摆位；输入 MOVE 后要求连续发布的 `/lowstate` 解码遥控数据中立至少 0.5 秒，并检查停稳。`/wirelesscontroller` 在本机间歇发布，不作为必需心跳；其最近活动/异常事件仍会阻止执行。未满足中立条件只重新提示，不发送运动指令。
3. 每次保存 before/after 图像、控制指令、实测局部 dx/dy/yaw、累计路程/转角、图像时间间隔。全程另存录像和里程计 CSV。脚本不自动回到原位置；可在下一次确认前人工调整位置，每次重新记录起始姿态。
4. 验证所有候选，保存 `calibration.json`。重复试验相对平均值的最大平移偏差须 ≤0.04 m，角度偏差须 ≤π/60。非零平移平均终点距离须 ≥0.2 m；非零转角平均幅度须 ≥π/12；等效圆弧路程与名义值相差 ≤0.05 m，转角与名义值相差 ≤π/36；旋转漂移和直行偏航也有限制。任何失败试验均不被静默剔除，未完成或不合格文件 `ready=false`，导航拒绝使用。
5. 最终重复试验不再调整指令。失败时保留所有原始记录，退出并停止；不会自动重试不合格动作。可通过 `--resume 旧目录/calibration.json` 将已验证样本读入新的输出目录；完整候选跳过，未完成候选保留已有重复样本及冻结指令，仅补缺失次数。原目录不被修改。

标定与导航的硬限制统一为累计路程 0.70 m / 累计转角 π/3（60°），运动时提前预留 0.015 m / 4° 刹停，约在 0.685 m / 56° 达到预留阈值。名义候选仍为 0.25/0.50 m、π/12/π/4；硬限制不是目标幅度，也不能保证实际零超调。达到预留阈值的有效样本可以标定；真正越过硬限制或反馈失败则拒绝。导航模型使用实测平均终点，不把提前停止的结果伪装成名义终点。π/12（15°）名义候选的试调目标为 0.1π（18°），试动作至少达到 17° 才冻结指令进入重复测量，以避免单次刚达标、平均值又落到下限以下。最终仍要求三次平均值 ≥15°，与名义值相差 ≤5°，重复性偏差 ≤3°，实际值如实保存。

标定只能刻画当前地面、步态和载荷下的重复性；保留了避障控制，受障碍影响的样本可能不能代表自由运动。应在空旷区域采集，检查 before/after 与轨迹；不声称自动识别了所有避障干预。

## 机载命令

以下命令在机载交互式终端执行。每次运行使用新的输出目录。代码同步后执行：

```bash
cd /home/unitree/ndj/code/OpenNWM
PY=/home/unitree/ndj/envs/opennwm/bin/python
RUN_BASE=$(mktemp -d /home/unitree/ndj/visualizations/opennwm/fixed15-XXXXXX)
echo "$RUN_BASE"

# 可选：查看计划，不连接机器人。
"$PY" -B deploy/jetson_calibrate_actions.py plan --output "$RUN_BASE/plan"

# 实际离线标定：每个试动作/重复动作均逐次输入 MOVE NNN。
"$PY" -B deploy/jetson_calibrate_actions.py collect \
  --interface eth10 --repeats 3 --max-tuning-rounds 8 \
  --output "$RUN_BASE/calibration"
```

成功完成后，沿用相同终端中的 `RUN_BASE`（换终端请设为上一步打印的真实目录），先做只读预览：

```bash
"$PY" -B deploy/jetson_navigate.py preview \
  --model opennwm-dmd --sampler dmd --steps 4 --batch-size 15 \
  --action-calibration "$RUN_BASE/calibration/calibration.json" \
  --reference /home/unitree/ndj/data/go2-workspace/datasets/episode_20261019T041125_81f8fa8c \
  --reference-seconds 30 --target-rule either --interface eth10 \
  --output "$RUN_BASE/preview"

"$PY" -B deploy/jetson_navigate.py navigate \
  --model opennwm-dmd --sampler dmd --steps 4 --batch-size 15 \
  --action-calibration "$RUN_BASE/calibration/calibration.json" \
  --reference /home/unitree/ndj/data/go2-workspace/datasets/episode_20261019T041125_81f8fa8c \
  --reference-seconds 30 --target-rule either --interface eth10 \
  --output "$RUN_BASE/navigation"
```

## 导航行为

- 每轮排名包含 15 个候选。静止直接使用当前真实图像与目标的 LPIPS；仅对 14 个运动候选批量推理和解码。默认 batch size 14；兼容旧参数 15，但实际仍只推理 14 个。静止胜出时显示当前真实图像，并标记图像来源。
- 只读取冻结的标定指令和实测动作条件；执行后保存偏差，绝不更新尺度、指令或补偿。读取时重新检查原始样本与控制参数，不只相信 ready 标记或可编辑的平均值。
- 静止候选直接使用当前真实图像与目标的 LPIPS 参与排名，不参与模型推理；选中后不发送运动，也不自动宣布到达。导航仍保留 START、每步 MOVE、arrived 与 q。
- 所有候选（包括静止）统一使用 0.5 秒预测条件：2 帧 / 128 = `k=0.015625`。标定的 before/after 图像平均间隔只作测量记录，不参与模型时间条件计算。标定的动作位移、速度和执行持续时间不变。
- 四张历史图仍沿用现有的动作后观测顺序。此更新未证明历史采样、0.25 单位及时间条件在该权重上达到最佳预测精度；需用实际标定图像/导航结果进一步检验。
- 摄像头、遥控、里程计或录制故障，超时、硬限位和中断仍会停止。标定脚本和导航不可同时持有运动锁。

本次代码验证不执行机器人运动，也不会生成冒充实测的可用标定文件。GPU 冒烟测试使用独立合成图像和测试夹具，验证 14 行运动动作/时间条件批量推理，以及静止直接评分的行为。

## 遥控摆位与中断续跑

仅标定模式允许等待阶段的遥控操作。确认后，从采集起始位姿到执行、停稳、最终图像记录完成，遥控接管仍锁存故障并停止；松开遥控器不清除执行期间的故障。普通导航的接管保护不变。

如果 MOVE 输入时 `/lowstate` 遥控数据尚未连续中立 0.5 秒，或最近 0.5 秒收到 `/wirelesscontroller` 活动/异常事件，程序只提示松开并重新输入同一个 MOVE；不会在你松开后自动启动。每次重新摆位后都会使用新起始位姿测量。

针对 fixed15-JqJcaF 中断结果，保留 3 个静止样本和 1 个 forward_25 样本，在机载终端执行：

```bash
cd /home/unitree/ndj/code/OpenNWM
RUN_BASE=$(mktemp -d /home/unitree/ndj/visualizations/opennwm/fixed15-resume-XXXXXX)
echo "$RUN_BASE"
/home/unitree/ndj/envs/opennwm/bin/python -B deploy/jetson_calibrate_actions.py collect \
  --interface eth10 --repeats 3 --max-tuning-rounds 8 \
  --resume /home/unitree/ndj/visualizations/opennwm/fixed15-JqJcaF/calibration/calibration.json \
  --output "$RUN_BASE/calibration"
```

新的试验编号从 MOVE 001 开始，按屏幕当前编号确认；此次首个试验是 forward_25 的第 2 次重复。完成后的导航使用新目录里的 calibration.json。续跑会重新校验原始样本，不会把原本 failed、越界、缺失反馈或不同指令采集的样本变成有效样本。

试调次数耗尽时，程序保存下一次待验证指令并以 `needs_more_pilots` 暂停，不把未达标结果计入样本，也不再输出重复 traceback。恢复运行仍需逐次 MOVE，已有重复样本的冻结指令不改变。

旧 0.50 m / 45° 配置须通过 `collect --resume` 迁移到新输出目录。按时完成的有效重复样本保留；若某候选旧样本曾因刹停预留提前结束，该候选整组重新采集，因为放宽限位后相同指令的终点可能变化。失败样本不会被升级成有效样本。导航拒绝直接读取旧限位版本，使用新采集完成的 calibration.json。

## 指定候选重新采集

若同一指令的重复结果未通过质量检查，程序以 `needs_recollection` 保存进度暂停，并显示实际最大平移/转角偏差。用 `--resume 旧目录/calibration.json --redo-candidate left_arc_50_pi12` 在新输出目录重测该候选；可重复指定参数以重测多个候选。原始文件保持不变，新运行只清空指定候选的样本，保留其他合格候选及冻结指令。被重测候选重新做试动作和完整重复测量，不能挑选较好的旧样本混入以绕过重复性检查。旧目录中其他不合格样本仍会使续跑校验失败。

对指定重采的候选，若旧的至少 3 次同指令重复测量均为正常定时完成、重复性合格，但平均距离/转角偏离名义值，新运行会先根据这组平均值生成下一次待验证指令，保存 `recollection-adjustments.json`。每轴指令总量的变化仍限制在 0.8–1.2。旧样本不会混入新重复组。已知避障干预、失败、刹停预留停止、指令不一致或重复性不合格的数据不会用于此调整。

原地转向试调支持历史区间收敛：同一转速下，若已有低于和高于目标角度的正常定时试验，则在最窄时长区间内插值（限制在区间中央 25%–75%），避免单点比例调整反复过冲。小角度试调目标仍为 18°。失败、触及刹停预留、已知避障干预、方向错误或平移漂移超过 0.04 m 的试验不参与区间计算；这并不意味着能可靠检测所有避障干预。

`pilot-history.json` 保存试调历史，`--resume` 自动恢复它及源目录中的 pilot execution.json。历史只用于未冻结候选的指令试调，不计入合格重复样本；显式 `--redo-candidate` 清除该候选的历史区间。每次实际执行仍需 MOVE 确认，新的指令只有实际验证后才可进入重复采集。该方法减少比例模型导致的震荡，不保证机器人响应必然收敛。

## 当前导航默认值（2026-10-20）

导航和预览默认读取脚本旁的 `default-action-calibration.json`，来自机载 `fixed15-resume-FDXiWv/calibration/calibration.json`（15 个动作、45 个合格样本）。每次启动仍重新校验完整样本；可用 `--action-calibration` 显式覆盖。OpenNWM 和 NWM 共用此动作标定。

终端默认简洁输出：目标/步数、动作与标定位移、当前图像与目标 LPIPS、执行实测结果和人工到达确认。详细诊断及 traceback 保存在输出目录的 `navigation.log`；`--verbose` 恢复详细终端日志。START / MOVE / arrived / q 交互和运动保护保持有效。

导航执行确认仅需输入 `MOVE`（不带编号）；步号仍用于日志和文件记录。每次成功完成动作后，在到达确认提示之前计算、显示并保存实际图像与当前目标的 LPIPS，输入 arrived 或 q 也不会漏存该分数。标定采集仍使用其原有 MOVE 编号。

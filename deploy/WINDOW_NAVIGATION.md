# 固定时间采样与前向窗口导航

两个独立入口：`window_navigate.py` 为手动，`auto_window_navigate.py` 为自动。原先两个导航入口的目标选取和交互保持不变。

## 目标与动作联合选择

- 默认从参考起点开始，每隔 0.5 秒选取一个目标帧；取每个时间刻度处或之后的第一帧，不重复同一帧。第一帧用于起点对齐，首次窗口中心为其后第一张采样帧。
- 默认 10 帧窗口：当前帧及其后最多 9 帧；尾部不足 10 帧时缩短，不包含当前索引之前的目标。
- 在所有通过动作约束的候选与所有窗口目标的组合中，选择匹配距离最低的一对。不添加进度惩罚或到达阈值。目标中心更新为匹配帧，只能保持或前移。
- 每个运动候选每轮只推理、VAE 解码一次，预测图像复用于全部目标评分；静止使用当前真实图像，不推理。所有模型运动时间条件仍是 0.5 秒、k=0.015625，与目标采样间隔独立。
- 自动模式输入一次 START，运动后停稳、获取新图像、输出实际图像与本轮匹配目标的 LPIPS，立即继续。静止胜出时将匹配目标判为到达，窗口中心移动到其后一帧；匹配最后一帧且静止胜出才结束。
- 手动模式逐轮输入 MOVE，运动后回车继续；arrived 人工确认本轮匹配目标并移到其后一帧，q 退出。静止不会在手动模式自动确认到达。
- 不允许回滚；同一目标的静止动作在近似同一位置（3 cm 内）及朝向（5° 内）首次重复时，立即识别为循环，包括跨多步返回的循环。仅在本次选择中排除所有目标的 stationary，复用评分和预测图像选择最低匹配距离的合规运动候选；下一轮恢复静止候选资格。保留循环历史，若运动后仍回到旧静止状态，则重新触发一次。无需连续重复多次，不增加到达阈值或进度惩罚。无合规运动候选时停止报错。
- 相似场景仍可能导致停滞或新的循环，默认最多 50 次推理（含静止），可用 --max-actions 修改。到达记录是所选目标的策略/人工确认，不代表所有中间帧均逐个到达。
- 原地反向转动约束、标定、避障调用、遥控接管与异常停止共用现有实现；左右弧线允许。自动模式 Ctrl+C 停止。

## 机载命令

```bash
cd /home/unitree/ndj/code/OpenNWM
PY=/home/unitree/ndj/envs/opennwm/bin/python
"$PY" -B deploy/window_navigate.py navigate \
  --model opennwm-dmd --sampler dmd --steps 4 --batch-size 14 \
  --reference /home/unitree/ndj/data/go2-workspace/datasets/episode_20261019T041125_81f8fa8c \
  --reference-seconds 30 --target-interval 0.5 --target-window 10 --interface eth10 \
  --output /home/unitree/ndj/data/go2-workspace/window-manual-$(date +%Y%m%d-%H%M%S)
```

自动模式将脚本名改为 `deploy/auto_window_navigate.py`，输出目录改为新的唯一目录。支持 `prepare` 只生成参考目标和 `preview` 只推理不运动；NWM DMD 使用 `--model nwm-dmd`。时间窗口模式不使用旧的 `--target-rule` 距离/角度规则。

每轮 `target-window.json` 记录推理前中心和窗口成员，`selection.json` 记录所有候选×目标的匹配距离排名与选中目标，`target.json` 记录最终匹配目标序号及 `reference/` 文件路径。终端显示目标索引变化、动作、最佳匹配距离、执行后实际 LPIPS，以及约束触发/解除。`reference.json` 和 `reference.png` 记录采样结果。

循环触发时终端显示本轮排除全部 stationary 和重选结果；`selection.json` 的 `loop_avoidance` 记录触发来源及单次屏蔽范围，`selected_score` 为重选结果分数，`ranking` 保留全部原始评分。

## 匹配指标

默认 `--matching-metric lpips` 保留 LPIPS。添加 `--matching-metric dinov2` 使用 DINOv2-S/14 的归一化 CLS 特征余弦距离（1 − cosine，越小越接近），静止观测和运动预测使用相同评分。图像使用导航模型相同的 224×224 裁剪，再做 ImageNet 标准化；每批预测特征只提取一次，复用于整个窗口。

模型默认从 `--model-root` 下的 `planning/dinov2-small` 离线加载，也可用 `--dinov2-model /path/to/model` 指定本地 Hugging Face 模型目录。模型缺失时报错，不会静默回退或在导航中下载。需要 transformers 的 DINOv2 支持。

`selection.json` 的 `matching_metric`、`selected_score`、`current_goal_score` 和排名中的 `score` 记录选用指标；`lpips` 或 `dinov2` 字段标明分数类型。现场变化检测和执行后的 LPIPS 诊断保持原有定义，DINOv2 分数不使用 LPIPS 安全阈值。

## 自动提案失效恢复

执行前若位置或画面变化导致提案失效，立即发送停止指令，丢弃旧提案，恢复本轮选择前的目标索引。确认静止后，采集新帧，要求至少 0.4 秒内相对稳定期起始图像的 LPIPS ≤ 0.05，且位姿和速度检查通过；期间变化会重新开始稳定计时。最多等待 10 秒。

恢复后使用最新稳定帧重建四帧模型上下文，保留动作执行记录、转向约束和循环检测历史，自动重新推理。旧动作不会执行，也不因被拒绝的匹配推进目标。连续 3 次提案失效则停止；成功通过下一次执行前检查后清零计数。重试仍计入 `--max-actions`。相机故障、遥控接管及其他异常继续停止退出，不属于自动恢复范围。手动模式仍需输入 `replan`。

## 批量匹配与精简记录

所有目标帧在导航前统一保存于 `reference/`，`reference.json` 保留序号和源帧映射。导航期间持续保存相机原始分辨率的 `navigation.mkv`（H.264，8 Mbps）、`video-timestamps.csv`、`trajectory.csv`。每轮只保存一张原始分辨率 `actual-after.png`：运动执行后采集新帧；静止轮使用执行前检查通过的最新帧。被拒绝的提案未执行，不产生动作后图片。

不再重复写入历史图、目标图、224px 输入图、预测图和决策拼图；实时监看仍在内存中显示这些图像。`target.json` 记录本轮选中的目标序号、参考文件和源帧号；`selection.json`、执行、拒绝和恢复等 JSON 记录保留，用于区分执行成功和被取消的提案。

LPIPS 把候选×目标展平为图像对批量评分，默认 `--matching-batch-size 140`，覆盖 14 个候选×10 个目标；更大的窗口自动分批，可减小该参数限制显存。批量计算存在浮点舍入差异，极接近的分数可能交换顺序。DINOv2 每批提取候选特征后，以矩阵乘法计算全部目标分数；窗口间缓存目标特征，内容改变时重新编码。预测结果依赖模型前向和解码，匹配阶段须在预测完成后执行。

## 历史帧更新模式

无线入口可加 `--history-update round`：每轮结束采集一张真实相机帧，执行运动时等动作结束后采集；静止、人工确认到达或未确认运动的轮次也更新。每轮只追加一次，最近四帧作为下轮输入，不足四帧向左补齐。默认 `--history-update action` 保持仅运动结束后追加。异常提案恢复仍重建稳定历史。不会把模型预测图像写入历史。

推理默认 batch size 为7（含静止）；LPIPS匹配批量自动为 batch size × 本轮目标数，无需单独设置。

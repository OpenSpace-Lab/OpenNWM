# 无线远程导航

在本机运行 `deploy/remote_navigate.py`，通过 SSH 在机载启动窗口导航，并将只读可视化页面转发到本机。导航推理和运动控制仍在机载执行，内部控制接口仍为 `eth10`。

## 已验证连接

2026-10-23：机载 USB TP-Link AIC8800D80 网卡连接 `office`，接口 `wlan0`，DHCP 地址 `192.168.14.129`；本机 `wlo1` 地址 `192.168.14.103`。临时断开本机有线网络配置后，无线 SSH 和真实摄像头页面均通过验证。完整模型预览完成，没有执行运动。可以拔掉本机与机器狗之间的网线；保留机载与机器人控制器之间的内部网络连接。IP 可能随 DHCP 分配变化。

## 本机启动

在 OpenNWM 项目根目录执行：

```bash
python3 deploy/remote_navigate.py 192.168.14.129 auto \
  --model opennwm-dmd --sampler dmd --steps 4 \
  --batch-size 14 --matching-batch-size 140 \
  --reference-seconds 30 --target-interval 0.5 --target-window 10 \
  --matching-metric lpips --max-actions 50 --seed 0
```

按 SSH 提示输入密码。对齐参考起点后按机载终端提示输入 START；Ctrl+C 停止，并保留现场遥控器接管。保持终端连接，在本机浏览器打开 http://127.0.0.1:8765。页面只在导航服务运行期间开放。

- `auto`：运行机载 `auto_window_navigate.py navigate`。
- `manual`：运行机载 `window_navigate.py navigate`，逐轮确认动作。
- `preview`：只推理，不执行运动。
- `check`：检查 SSH、机载网络和入口文件。
- `monitor`：只建立页面隧道，导航需已通过其他终端启动。
- `--dry-run`：只打印命令，不连接或启动。
- `--nav-help`：读取机载入口的最新帮助，不启动导航。

## 参数

直接在命令中修改参数；每次启动生效，运行中修改需停止后重启。所有模型、参考数据和输出路径均是机载路径。

| 参数 | 默认值及含义 |
| --- | --- |
| `--batch-size` | 14，最多 14 个运动候选一起推理；约束过滤后可能少于 14。兼容值 15 也只推理 14 个运动候选。静止使用当前真实画面直接评分。 |
| `--matching-batch-size` | 140，LPIPS 每批最多比较 140 对图像，默认对应 14×10；与推理 batch 不同。 |
| `--model` | opennwm-dmd，也支持 nwm-dmd、opennwm、nwm-b。 |
| `--sampler`、`--steps` | 未指定时由机载按模型选择默认值；DMD 默认 4 步。steps 是模型采样步数，不是行走步数。 |
| `--reference` | 默认机载 episode_20261019T041125_81f8fa8c 数据集。 |
| `--reference-seconds` | 30，使用参考轨迹前 30 秒。 |
| `--target-interval` | 0.5，参考目标帧采样间隔，不是运动指令时长。 |
| `--target-window` | 10，当前及后续参考目标窗口，至少 3。 |
| `--matching-metric` | lpips；可改 dinov2，要求机载存在相应模型。 |
| `--max-actions` | 50，提案/推理轮次上限，包括被取消的提案，有效范围 1–500。 |
| `--output` | 自动生成新的机载时间戳目录；显式指定时目录必须尚不存在。 |
| `--port` | 8765，本机页面端口。 |
| `--display-port` | 8765，机载页面端口；SSH 转发自动匹配。 |

## 网卡安装记录

驱动来源 https://github.com/radxa-pkg/aic8800 ，提交 `fce88024ed40279def034e203128e0892fdbcac3`。为 Jetson `5.10.104-tegra` 内核本机编译，并在 USB 匹配表和芯片匹配分支中补充 TP-Link `2357:014e`，映射到 AIC8800D80/D81 分支。设备最初为存储模式 `a69c:5721`，切换后为加载固件模式 `a69c:8d80`，加载固件后为 `2357:014e`。

机载源代码、安装脚本及构建日志：`/home/unitree/ndj/work/aic-usb-test`。模块：`/lib/modules/5.10.104-tegra/extra/go2-aic`。自动切换规则：`/etc/udev/rules.d/70-go2-aic-modeswitch.rules`。NetworkManager 连接 `go2-office` 已设置自动连接并关闭省电。重启/拔插自动恢复尚未实测，内核升级后需重编译。

短时无线测试：25 次 ping 无丢包，平均 36.387 ms，最大 125.953 ms。尚未验收长距离稳定性和运动时无线失联急停。

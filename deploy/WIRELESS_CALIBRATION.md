# 无线标定

先拔掉本机与机器狗之间的外接网线，机载内部 eth10 保留。机器狗置于平整空旷区域，用遥控器进入可行走状态。标定期间保持无线网卡、地面和负载与以后导航一致。

在本机 OpenNWM 根目录运行：

```bash
python3 deploy/remote_calibrate.py 192.168.14.129 collect --new-terminal
```

输入 SSH 密码。每次检查空间、松开遥控器，按终端提示输入 `MOVE 001` 等精确编号；`q` 或 Ctrl+C 退出。不会连续自动执行全部动作。看到明显后退、侧滑时停止并检查，不要仅为了完成采集继续确认。

默认从零开始，不读取旧有线标定：15 个候选，每个正式采样 3 次；运动候选先试调（默认最多 8 次），最少共 59 次观察。每次动作都要确认。正式样本须通过原有重复性、幅度及旋转漂移检查；不合格不会成为可用标定。本入口不新增实时 4 cm 漂移急停，仍使用现有总运动限位；旋转净漂移 4 cm 是标定质量检查，不能等同于实时停止保护。

程序检查并绑定本机 Wi-Fi 接口连接机载；机载运动控制继续使用内部 eth10。该检查不能证明网线已物理拔掉，需现场确认。SSH 保活不等于已验证的运动断网急停。

输出目录自动生成并打印。`calibration.json` 只有完整且合格时才为 `ready=true`；`status.json`、`calibration.log`、录像和逐次 before/after 记录都在该目录。不会替换默认标定文件。

可选参数：

```bash
# 只生成计划，不连接机器人运动接口
python3 deploy/remote_calibrate.py 192.168.14.129 plan

# 每个候选采集 5 次正式样本
python3 deploy/remote_calibrate.py 192.168.14.129 collect --repeats 5 --new-terminal

# 仅续跑同一无线配置下的未完成标定；自动写入新目录，原目录保留
python3 deploy/remote_calibrate.py 192.168.14.129 collect \
  --resume /home/unitree/ndj/visualizations/opennwm/实际旧目录/calibration.json --new-terminal
```

`--resume` 时 repeats 必须与源文件相同；如需重测某候选，可加 `--redo-candidate rotate_left_pi12`，不要用旧有线标定续跑来代替本次全新无线标定。

完成后，在本机启动导航并显式选择新标定文件（替换实际目录）：

```bash
python3 deploy/remote_navigate.py 192.168.14.129 auto --new-terminal \
  --action-calibration /home/unitree/ndj/visualizations/opennwm/实际新目录/calibration.json
```

标定本身不加载导航模型，也没有导航网页；日志和操作都在独立终端，录像与数据保存在机载。加 `--dry-run` 只看命令，`--help` 查看全部参数。

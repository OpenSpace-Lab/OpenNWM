#!/usr/bin/env python3
"""在本机启动机载导航；参数路径均指向机载电脑，--port 是本机端口。"""
import argparse
import datetime
import os
import re
import shlex

p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
p.add_argument('host', help='机载无线 IP，例如 192.168.14.129')
p.add_argument('mode', choices=['check', 'preview', 'manual', 'auto', 'monitor'], nargs='?', default='check')
p.add_argument('--port', type=int, default=8765, help='本机浏览器端口')
p.add_argument('--display-port', type=int, default=8765, help='机载可视化端口，自动匹配 SSH 转发')
p.add_argument('--dry-run', action='store_true', help='仅打印命令，不连接、不启动导航')
p.add_argument('--nav-help', action='store_true', help='读取机载导航脚本的最新帮助，不启动导航')
nav = p.add_argument_group('导航参数（发送给机载脚本）')
nav.add_argument('--model', choices=['opennwm', 'nwm-b', 'opennwm-dmd', 'nwm-dmd'], default='opennwm-dmd')
nav.add_argument('--sampler', choices=['ddpm', 'dpmpp_2m', 'ddim', 'dmd'])
nav.add_argument('--steps', type=int)
nav.add_argument('--batch-size', type=int, choices=[14, 15], default=14)
nav.add_argument('--reference', default='/home/unitree/ndj/data/go2-workspace/datasets/episode_20261019T041125_81f8fa8c')
nav.add_argument('--reference-seconds', type=float, default=30)
nav.add_argument('--target-interval', type=float, default=0.5)
nav.add_argument('--target-window', type=int, default=10)
nav.add_argument('--matching-metric', choices=['lpips', 'dinov2'], default='lpips')
nav.add_argument('--matching-batch-size', type=int, default=140)
nav.add_argument('--max-actions', type=int, default=50)
nav.add_argument('--seed', type=int, default=0)
nav.add_argument('--interface', default='eth10')
nav.add_argument('--output', help='机载输出目录，必须是新的目录；默认按时间生成')
for option in ['model-root', 'runtime-root', 'ros-setup', 'action-calibration', 'dinov2-model']:
    nav.add_argument('--' + option)
nav.add_argument('--ddim-eta', type=float)
nav.add_argument('--verbose', action='store_true')
a = p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.:-]*', a.host):
    p.error('host 必须为 IP 或主机名')
if not (1024 <= a.port <= 65535 and 1024 <= a.display_port <= 65535):
    p.error('本机和机载页面端口必须在 1024–65535 之间')
ssh = ['ssh', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=3', '-o', 'ServerAliveCountMax=3', '-o', 'ExitOnForwardFailure=yes']
dest = 'unitree@' + a.host
entry = 'auto_window_navigate.py' if a.mode == 'auto' else 'window_navigate.py'
base = ['/home/unitree/ndj/envs/opennwm/bin/python', '-B', '/home/unitree/ndj/code/OpenNWM/deploy/' + entry]
if a.nav_help:
    ssh += [dest, shlex.join(base + ['--help'])]
elif a.mode == 'check':
    cmd = 'hostname; ip -br addr; test -x /home/unitree/ndj/envs/opennwm/bin/python && test -f /home/unitree/ndj/code/OpenNWM/deploy/window_navigate.py && echo NAVIGATION_READY'
    ssh += [dest, cmd]
else:
    ssh += ['-L', f'127.0.0.1:{a.port}:127.0.0.1:{a.display_port}']
    print(f'监看地址：http://127.0.0.1:{a.port}（导航服务启动后可访问）', flush=True)
    if a.mode == 'monitor':
        ssh += ['-N', dest]
    else:
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        a.output = a.output or '/home/unitree/ndj/visualizations/opennwm/remote-' + a.mode + '-' + stamp
        # 不指定 sampler/steps 时让机载脚本按模型选默认值；DMD 为 4 步。
        command = base + ['preview' if a.mode == 'preview' else 'navigate']
        for action in nav._group_actions:
            value = getattr(a, action.dest)
            if value is None or value is False:
                continue
            command.append(action.option_strings[0])
            if value is not True:
                command.append(str(value))
        command += ['--display-port', str(a.display_port)]
        print('机载命令：' + shlex.join(command), flush=True)
        ssh += ['-tt', dest, 'exec ' + shlex.join(command)]
if a.dry_run:
    print('仅预览命令，未执行：\n' + shlex.join(ssh))
else:
    os.execvp('ssh', ssh)

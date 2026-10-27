#!/usr/bin/env python3
"""在本机通过 Wi-Fi SSH 运行机载标定，每个动作仍须输入 MOVE 编号。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('host', help='机载无线 IP，例如 192.168.14.129')
    p.add_argument('mode', choices=['plan', 'collect'], help='plan 只生成计划；collect 逐次确认采集')
    p.add_argument('--repeats', type=int, default=3, help='每个动作的正式重复次数，3–20')
    p.add_argument('--max-tuning-rounds', type=int, default=0, help='0：持续试调直到验收通过或手动退出；1–12：限制每组试调次数和重试组数')
    p.add_argument('--output', help='新的机载输出目录；默认自动生成')
    restart = p.add_mutually_exclusive_group()
    restart.add_argument('--resume', help='显式指定机载 calibration.json；默认自动续跑最近兼容无线标定')
    restart.add_argument('--fresh', action='store_true', help='从头开始，不读取任何旧进度')
    p.add_argument('--redo-candidate', action='append', default=[], help='配合 resume 重测指定候选')
    p.add_argument('--new-terminal', action='store_true', help='新开独立标定终端，退出后保留窗口')
    p.add_argument('--dry-run', action='store_true', help='只打印命令，不连接或采集')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.:-]*', a.host):
        p.error('host 必须是 IP 或主机名')
    if not 3 <= a.repeats <= 20 or not 0 <= a.max_tuning_rounds <= 12:
        p.error('repeats 必须为 3–20；max-tuning-rounds 必须为 0–12')
    if (a.resume or a.fresh or a.redo_candidate) and a.mode != 'collect':
        p.error('resume、fresh、redo-candidate 仅用于 collect')
    if a.redo_candidate and a.fresh:
        p.error('redo-candidate 不能与 fresh 同用')
    if a.new_terminal:
        terminal = shutil.which('gnome-terminal')
        if not terminal:
            p.error('未找到 gnome-terminal；请手动新开终端并去掉 --new-terminal')
        child = [sys.executable, str(Path(__file__).resolve())] + [v for v in sys.argv[1:] if v != '--new-terminal']
        shell = shlex.join(child) + '; result=$?; printf "\\n标定会话已结束（退出码 %s）。\\n" "$result"; exec bash'
        cmd = [terminal, '--title=OpenNWM 无线标定', '--working-directory=' + str(Path(__file__).resolve().parent.parent), '--', 'bash', '-c', shell]
        if a.dry_run:
            print(shlex.join(cmd)); return 0
        return subprocess.run(cmd).returncode
    # 仅检查本机通往机载的链路；机载控制器仍使用内部 eth10。
    try:
        address = socket.gethostbyname(a.host)
        route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', address], text=True))[0]
        interface = route['dev']
        if not Path('/sys/class/net', interface, 'wireless').exists():
            p.error('当前连接没有走本机 Wi-Fi；请使用机载无线 IP')
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError) as exc:
        p.error('无法确认无线链路：' + str(exc))
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = a.output or '/home/unitree/ndj/visualizations/opennwm/wifi-calibration-' + stamp
    remote = ['/home/unitree/ndj/envs/opennwm/bin/python', '-B',
              '/home/unitree/ndj/code/OpenNWM/deploy/jetson_calibrate_actions.py', a.mode,
              '--interface', 'eth10', '--repeats', str(a.repeats),
              '--max-tuning-rounds', str(a.max_tuning_rounds), '--output', output]
    if a.resume:
        remote += ['--resume', a.resume]
    elif a.mode == 'collect' and not a.fresh:
        remote += ['--auto-resume']
    for name in a.redo_candidate:
        remote += ['--redo-candidate', name]
    command = ['ssh', '-B', interface, '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=3',
               '-o', 'ServerAliveCountMax=3', '-tt', 'unitree@' + address, 'exec ' + shlex.join(remote)]
    print('无线接口：' + interface, flush=True)
    print('机载输出：' + output, flush=True)
    if a.mode == 'collect':
        print('进度：' + ('从头开始' if a.fresh else ('指定续跑文件' if a.resume else '自动续跑最近兼容无线标定；合格动作跳过')), flush=True)
    print('新标定文件：' + output + '/calibration.json', flush=True)
    print('首次无线标定请拔掉外接网线；每次输入终端显示的 MOVE 编号。q 或 Ctrl+C 退出。', flush=True)
    print('完成且 ready=true 后，导航加 --action-calibration ' + shlex.quote(output + '/calibration.json'), flush=True)
    if a.dry_run:
        print('未执行：' + shlex.join(command)); return 0
    return subprocess.run(command).returncode


if __name__ == '__main__':
    raise SystemExit(main())

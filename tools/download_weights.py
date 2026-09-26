#!/usr/bin/env python3
"""Download selected public weights; completed files are reused, partial downloads resume."""
import argparse
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

REVISION = 'f18fc3845ff80a4626c83f82882d250ff2030637'
BASE = f'https://modelscope.cn/models/LittleBoss/LAM-weights/resolve/{REVISION}/'
ASSETS = {
    'lam-action': ('NWM-LatentPT/weights/ActionOnlyLAM/step=100000.ckpt', 'lam-action-100000.ckpt'),
    'lam-pixel-action': ('NWM-LatentPT/weights/PixelActionLAM/step=100000.ckpt', 'lam-pixel-action-100000.ckpt'),
    'nwm-stage1': ('NWM-LatentPT/weights/pretrain_0200000.pth.tar', 'nwm-latentpt-200000.pth.tar'),
    'dino': ('dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth', 'dinov3_vitl16.pth'),
}


def download(url, destination):
    if destination.exists():
        print(f'Reusing {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        with partial.open('ab' if append else 'wb') as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        if response.length not in (None, 0):
            raise IOError(f'Download interrupted; rerun to resume {partial}')
    partial.replace(destination)
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', choices=ASSETS, action='append', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    for name in args.asset:
        path, filename = ASSETS[name]
        download(BASE + path, args.output_root.expanduser().resolve() / filename)


if __name__ == '__main__':
    main()

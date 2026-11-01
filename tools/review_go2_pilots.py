#!/usr/bin/env python3
"""Make unaltered-frame contact sheets for fixed, output-independent pilots."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_go2_recording import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='opennwm', choices=['opennwm', 'nwm'])
    parser.add_argument('--indices', help='Optional fixed pilot IDs from another horizon')
    args = parser.parse_args()
    windows = json.loads((args.evaluation / 'windows.json').read_text())
    pilot_ids = set(map(int, args.indices.split(','))) if args.indices else set(windows['pilot_window_ids'])
    rows = [row for row in windows['windows'] if row['id'] in pilot_ids]
    horizon = windows['horizon']
    sample_steps = np.linspace(0, horizon, 5).astype(int)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    width, cell_w, cell_h, row_h = 1280, 256, 192, 450
    sheet = Image.new('RGB', (width, 60 + row_h * len(rows)), '#102033')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f'{args.evaluation.parent.name} | {args.model} | {windows["goal_interval_s"]:g}s fixed pilots', font=title_font, fill='white')
    summaries = []
    for i, row in enumerate(rows):
        directory = args.evaluation / 'outputs' / args.model / f'{row["id"]:06d}'
        result = json.loads((directory / 'result.json').read_text())
        metrics = result['metrics']
        top = 60 + row_h * i
        draw.text((12, top), f'Window {row["id"]} | aligned {row["start_time_s"]:g}s | ATE {metrics["ate_m"]:.3f}m | RPE {metrics["rpe_m"]:.3f}m | endpoint {metrics["endpoint_m"]:.3f}m | ref distance {row["reference_distance_m"]:.3f}m', font=font, fill='white')
        for col, step in enumerate(sample_steps):
            gt = Image.open(row['frames'][step]).convert('RGB')
            w, h = gt.size
            crop_w = int(h * 4 / 3)
            gt = gt.crop(((w - crop_w)//2, 0, (w + crop_w)//2, h)).resize((cell_w, cell_h))
            pred = gt if step == 0 else Image.open(directory / 'predicted_frames' / f'{step-1:03d}.png').convert('RGB').resize((cell_w, cell_h))
            for tier, frame in enumerate((gt, pred)):
                y = top + 32 + tier * (cell_h + 20)
                sheet.paste(frame, (col * cell_w, y))
                draw.text((col * cell_w + 5, y + cell_h), f'{"GT" if tier == 0 else args.model} +{step/4:.2f}s', font=font, fill='#a9c4df')
        summaries.append(dict(id=row['id'], start_time_s=row['start_time_s'], clip_ids=row['clip_ids'],
            metrics=metrics, reference_distance_m=row['reference_distance_m'],
            endpoint_over_reference_distance=metrics['endpoint_m']/max(row['reference_distance_m'], .05),
            reference_stationary=row['stationary_reference']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    write_json(args.output.with_suffix('.json'), dict(evaluation=str(args.evaluation), model=args.model,
        horizon_seconds=windows['goal_interval_s'], pilots=summaries,
        display='Original model PNGs only resized; GT uses matching 4:3 central crop. No interpolation, frame blending or image generation.'))
    print(args.output)


if __name__ == '__main__':
    main()

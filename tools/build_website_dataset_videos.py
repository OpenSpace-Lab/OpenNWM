#!/usr/bin/env python3
"""Export a real consecutive-frame video for every website dataset card."""
from pathlib import Path
import json
import os
import shutil
import subprocess
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / 'website'
DATA = Path(os.environ.get('NWM_DATA_ROOT', '/file_system/nas/algorithm/dujun.nie/nwm/data'))
WORK = Path(os.environ.get('OPENNWM_DATASET_MEDIA_WORK', '/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20260930/datasets'))
MANIFEST = SITE / 'content/datasets.json'

# Reviewed continuous trajectories replace the old two/three-frame examples.
REPLACEMENTS = {
    'sacson': ('sacson', 'Dec-06-2022-bww8_00000000_0', 387, 48,
               'Through an indoor passage', 'A wheeled robot moves through a carpeted indoor passage.', ['indoor', 'robot']),
    'scand': ('scand', 'Bags_A_Jackal_Brackenridge_Greg_Sat_Nov_13_94_0', 406, 48,
              'Sharing the campus path', 'An outdoor robot navigates a campus walkway with people and bicycle racks.', ['urban', 'robot']),
    'tartan-drive': ('tartan', '20210826_heightmaps_1_20210826_42_2', 350, 48,
                    'Beyond paved roads', 'An off-road vehicle travels along a wooded gravel and mud track.', ['nature', 'robot']),
}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(WORK).free > 250_000_000
    doc = json.loads(MANIFEST.read_text())
    report = []
    for item in doc['datasets']:
        source = item['source']
        if item['id'] in REPLACEMENTS:
            folder, trajectory, total, center, title, description, environments = REPLACEMENTS[item['id']]
            source.update(trajectory=trajectory, trajectory_frame_count=total, image_frame=center,
                          relative_path=f'{folder}/{trajectory}/{center}.jpg')
            item.update(title=title, description=description, environments=environments)
        if item.get('video') and (SITE / item['video']).is_file():
            print(f'REUSE {item["id"]}: {item["video"]}', flush=True)
            continue
        total = source['trajectory_frame_count']
        count = min(96, total)
        first = max(0, min(source['image_frame'] - count // 2, total - count))
        directory = (DATA / source['relative_path']).parent
        paths = [directory / f'{i}.jpg' for i in range(first, first + count)]
        assert count >= 24 and all(p.is_file() for p in paths), (item['id'], paths)
        fps = 12 if count >= 60 else 8
        source['image_frame'] = first + count // 2
        source['relative_path'] = str((directory / f'{source["image_frame"]}.jpg').relative_to(DATA))
        with Image.open(DATA / source['relative_path']) as image:
            source['original_image_size'] = list(image.size)
            ImageOps.fit(image.convert('RGB'), (480, 270), method=Image.Resampling.LANCZOS).save(SITE / item['image'], quality=82)
        name = f'{item["id"]}.mp4'
        target = WORK / name
        command = ['/usr/bin/ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
                   '-framerate', str(fps), '-start_number', str(first), '-i', str(directory / '%d.jpg'),
                   '-frames:v', str(count), '-vf', 'scale=480:270:force_original_aspect_ratio=increase,crop=480:270,setsar=1',
                   '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '24', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-threads', '2', str(target)]
        with (WORK / f'{item["id"]}.log').open('w') as log:
            print(f'ENCODE {item["id"]}: {count} source frames at {fps} FPS', flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['/usr/bin/ffmpeg', '-v', 'error', '-i', str(target), '-f', 'null', '-'], check=True)
        shutil.copy2(target, SITE / 'assets/datasets' / name)
        item.update(video=f'assets/datasets/{name}', video_width=480, video_height=270, video_duration_seconds=count / fps)
        source['video'] = dict(first_frame=first, last_frame=first + count - 1, frame_step=1, playback_fps=fps,
                               note='Consecutive stored dataset frames, without synthetic motion or frame interpolation. Playback rate is editorial, not original capture timing.')
        report.append(dict(id=item['id'], frames=count, fps=fps, bytes=target.stat().st_size))
    assert all(item.get('video') and (SITE / item['video']).is_file() for item in doc['datasets'])
    doc.update(version=2, description='Continuous video examples from 15 NavAnywhere sources and 4 action-labeled navigation sources. These are source-data videos, not model predictions.')
    MANIFEST.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    (WORK / 'validation.json').write_text(json.dumps(dict(totalVideos=len(doc['datasets']), newVideos=report), indent=2) + '\n')
    print(f'COMPLETE: all {len(doc["datasets"])} dataset cards have source videos', flush=True)


if __name__ == '__main__':
    main()

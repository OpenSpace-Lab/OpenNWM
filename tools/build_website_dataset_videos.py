#!/usr/bin/env python3
"""Export a real consecutive-frame video for every website dataset card."""
from pathlib import Path
import argparse
from fractions import Fraction
import json
import os
import shutil
import subprocess
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / 'website'
DATA = Path(os.environ.get('NWM_DATA_ROOT', '/file_system/nas/algorithm/dujun.nie/nwm/data'))
WORK = Path(os.environ.get('OPENNWM_DATASET_MEDIA_WORK', '/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261029/dataset_timing_v2'))
MANIFEST = SITE / 'content/datasets.json'
NAS = '/file_system/nas/algorithm/dujun.nie/nwm'
CODE = '/file_system/vepfs/algorithm/dujun.nie/code'
NAV_TIMING = f'{NAS}/extract_navanywhere_frames.sh:106-119'
BOTANIC_VIDEO = Path('/file_system/nas/algorithm/dujun.nie/NavAnywhere/BotanicGarden/1005_00_VLIO/dalsa_color__left__image_raw.mp4')

# Rates refer to the stored frame sequence, not the camera's capture FPS.
# Bag preprocessing is nominally 4 Hz; its per-frame timestamps were discarded.
TIMING = {
    key: dict(fps=4, status='source_video', evidence=[NAV_TIMING])
    for key in ('citywalker', 'dl3dv', 'krishnacam', 'lavn', 'rover',
                'realestate10k', 'sanpo', 'walking-tours', 'i2nav-robot', 'ub-visiogeoloc')
}
TIMING.update({
    'botanic-garden': dict(fps=4, status='source_video', evidence=[str(BOTANIC_VIDEO)],
                           input_video=str(BOTANIC_VIDEO)),
    'casia-nav': dict(fps=4, status='source_video', evidence=[f'{NAS}/extract_casia_nav_frames.sh:39-41']),
    'great-outdoors': dict(fps=4, status='source_video', evidence=[f'{NAS}/extract_great_outdoors_frames.sh:57-59']),
    'ego4d': dict(fps=4, status='source_video', evidence=[f'{NAS}/logs/ego4d_frame_extraction/progress.log:4,901']),
    'egowalk': dict(fps=40, status='source_video', evidence=[
        f'{NAS}/resample_egowalk_40fps.sh:5', f'{NAS}/logs/egowalk_40fps/progress.log:94,241']),
    'sacson': dict(fps=4, status='nominal_sampling', evidence=[f'{CODE}/nwm/scripts/huron_public_pipeline.py:262']),
    'scand': dict(fps=4, status='nominal_sampling', evidence=[f'{CODE}/nwm/scripts/scand_hf_pipeline.py:308']),
    'tartan-drive': dict(fps=4, status='nominal_sampling', evidence=[f'{CODE}/nwm/scripts/tartandrive_pipeline.py:261']),
    'recon': dict(fps=4, status='display_only', evidence=[
        f'{NAS}/downloads/recon_raw/recon_release/jackal_2019-09-09-18-46-15_12_r01.hdf5'],
        limitation='The source has 108 images but 110 aligning_times values with irregular intervals; the image-to-timestamp mapping is unverified. 4 FPS is a display rate, not a claim of original speed.'),
})

# Reviewed continuous trajectories replace the old two/three-frame examples.
REPLACEMENTS = {
    'sacson': ('sacson', 'Dec-06-2022-bww8_00000000_0', 387, 48,
               'Through an indoor passage', 'A wheeled robot moves through a carpeted indoor passage.', ['indoor', 'robot']),
    'scand': ('scand', 'Bags_A_Jackal_Brackenridge_Greg_Sat_Nov_13_94_0', 406, 48,
              'Sharing the campus path', 'An outdoor robot navigates a campus walkway with people and bicycle racks.', ['urban', 'robot']),
    'tartan-drive': ('tartan', '20210826_heightmaps_1_20210826_42_2', 350, 48,
                    'Beyond paved roads', 'An off-road vehicle travels along a wooded gravel and mud track.', ['nature', 'robot']),
}


def public_metadata(value):
    """Keep source identities while excluding machine-local storage prefixes."""
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, str):
        for prefix in ('/file_system/nas/algorithm/dujun.nie/',
                       '/file_system/vepfs/algorithm/dujun.nie/code/'):
            if value.startswith(prefix):
                return value[len(prefix):]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Re-encode even if the timing-v2 asset exists.')
    args = parser.parse_args()
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
        timing = TIMING[item['id']]
        fps = timing['fps']
        total = source['trajectory_frame_count']
        count = min(320 if item['id'] == 'egowalk' else 96, total)
        first = max(0, min(source['image_frame'] - count // 2, total - count))
        directory = (DATA / source['relative_path']).parent
        paths = [directory / f'{i}.jpg' for i in range(first, first + count)]
        assert count >= 24 and all(p.is_file() for p in paths), (item['id'], paths)
        source['image_frame'] = first + count // 2
        source['relative_path'] = str((directory / f'{source["image_frame"]}.jpg').relative_to(DATA))
        with Image.open(DATA / source['relative_path']) as image:
            source['original_image_size'] = list(image.size)
            ImageOps.fit(image.convert('RGB'), (480, 270), method=Image.Resampling.LANCZOS).save(SITE / item['image'], quality=82)
        name = f'{item["id"]}-timing-v2.mp4'
        target = WORK / name
        published = SITE / 'assets/datasets' / name
        # BotanicGarden is cut directly from its original video timeline so the
        # playback timing does not depend on an inferred extraction rate.
        if timing.get('input_video'):
            input_args = ['-ss', str(first / fps), '-i', timing['input_video']]
            filters = f'fps={fps},'
        else:
            input_args = ['-framerate', str(fps), '-start_number', str(first), '-i', str(directory / '%d.jpg')]
            filters = ''
        command = ['/usr/bin/ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
                   *input_args,
                   '-frames:v', str(count), '-vf', filters + 'scale=480:270:force_original_aspect_ratio=increase,crop=480:270,setsar=1',
                   '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '24', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-threads', '2', str(target)]
        if published.is_file() and not args.force:
            print(f'REUSE {item["id"]}: {published}', flush=True)
            shutil.copy2(published, target)
        else:
            with (WORK / f'{item["id"]}.log').open('w') as log:
                print(f'ENCODE {item["id"]}: {count} source frames at {fps} FPS', flush=True)
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['/usr/bin/ffmpeg', '-v', 'error', '-i', str(target), '-f', 'null', '-'], check=True)
        probe = json.loads(subprocess.check_output([
            '/usr/bin/ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=avg_frame_rate,nb_frames,duration', '-of', 'json', str(target)], text=True))['streams'][0]
        assert float(Fraction(probe['avg_frame_rate'])) == fps, (item['id'], probe)
        assert int(probe['nb_frames']) == count, (item['id'], probe)
        assert abs(float(probe['duration']) - count / fps) < .01, (item['id'], probe)
        shutil.copy2(target, published)
        item.update(video=f'assets/datasets/{name}', video_width=480, video_height=270, video_duration_seconds=count / fps)
        notes = {
            'source_video': 'Playback preserves the source video timeline at its documented sampling rate. No motion interpolation.',
            'nominal_sampling': 'Playback uses the documented nominal 4 Hz preprocessing cadence. Exact per-frame capture timestamps are unavailable.',
            'display_only': timing.get('limitation', ''),
        }
        source['video'] = dict(first_frame=first, last_frame=first + count - 1, frame_step=1, playback_fps=fps,
                               timing_status=timing['status'], timing_evidence=timing['evidence'], note=notes[timing['status']])
        if timing['status'] != 'display_only':
            source['video'].update(source_sampling_fps=fps, playback_speed=1)
        if timing.get('input_video'):
            source['video'].update(input_video=timing['input_video'], source_start_seconds=first / fps,
                                   note='Sampled directly from the source video at 4 FPS; the original video timeline is preserved. No motion interpolation.')
        report.append(dict(id=item['id'], frames=count, fps=fps, duration_seconds=float(probe['duration']),
                           timing_status=timing['status'], bytes=target.stat().st_size))
    assert all(item.get('video') and (SITE / item['video']).is_file() for item in doc['datasets'])
    doc.update(version=3, description='Continuous video examples from 15 NavAnywhere sources and 4 action-labeled navigation sources. These are source-data videos, not model predictions.',
               playback_note='NavAnywhere clips preserve source-video time. SACSoN, SCAND, and TartanDrive use their nominal 4 Hz preprocessing cadence. RECON is shown at 4 FPS because frame timestamps could not be verified.',
               hero_video=next(item['video'] for item in doc['datasets'] if item['id'] == 'walking-tours'))
    MANIFEST.write_text(json.dumps(public_metadata(doc), ensure_ascii=False, indent=2) + '\n')
    (WORK / 'validation.json').write_text(json.dumps(dict(totalVideos=len(doc['datasets']), newVideos=report), indent=2) + '\n')
    print(f'COMPLETE: all {len(doc["datasets"])} dataset cards have source videos', flush=True)


if __name__ == '__main__':
    main()

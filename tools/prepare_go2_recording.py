#!/usr/bin/env python3
"""Preserve, align and clip the 2026-10-26 Go2 recording on its host clock.

The visual offset is a manually reviewed event match, not a hardware sync
calibration. Run with ``conda run --no-capture-output -n nwm python ...``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path('/file_system/nas/algorithm/dujun.nie/opennwm_go2_20261026')
HDR_FFMPEG = '/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/fastwam/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
EPISODE = 'episode_20261026T091121_148a01b5'
START, END, FPS, THIRD_OFFSET = 1.0, 173.0, 15, 3.15
CLIPS = [
    ('office_exit', 'Office exit', 16.0),
    ('corridor', 'Corridor traversal', 40.0),
    ('corner', 'Left corner', 58.0),
    ('foyer_exit', 'Foyer exit', 76.0),
    ('glass_door', 'Glass doorway', 110.0),
    ('turnaround', 'Turnaround', 132.0),
    ('return', 'Return through office', 156.0),
]


def command(args):
    print('COMMAND:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def nearest_indices(times, targets):
    right = np.searchsorted(times, targets).clip(1, len(times) - 1)
    left = right - 1
    return np.where(np.abs(times[left] - targets) < np.abs(times[right] - targets), left, right)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root):
    source = root / 'source' / EPISODE
    original_manifest = json.loads((source / 'manifest.json').read_text())
    poses = pd.read_csv(source / 'aligned_pose.csv')
    host = (poses.timeline_ns.to_numpy() - poses.timeline_ns.iloc[0]) * 1e-9
    output_t = np.arange(round((END - START) * FPS)) / FPS
    target_host = output_t + START
    ix = nearest_indices(host, target_host)
    timeline = pd.DataFrame({
        'output_frame_index': np.arange(len(output_t)),
        'aligned_time_s': output_t,
        'source_host_s': target_host,
        'source_frame_index': poses.frame_index.to_numpy()[ix],
        'source_video_s': poses.frame_index.to_numpy()[ix] / 15.0,
        'source_frame_host_s': host[ix],
        'nearest_frame_time_error_s': host[ix] - target_host,
        'third_source_s': target_host + THIRD_OFFSET,
        'pair_valid': poses.pair_valid.to_numpy()[ix],
    })
    # These interpolated poses are provided for visualization. Evaluation must
    # still honor the original pair_valid mask when constructing each window.
    valid = poses.pair_valid.fillna(False).astype(bool).to_numpy()
    for key in ['x', 'y', 'z', 'yaw', 'vx', 'vy', 'yaw_speed']:
        values = poses[key].to_numpy()
        if key == 'yaw':
            values = np.unwrap(values[valid])
        else:
            values = values[valid]
        timeline[key] = np.interp(target_host, host[valid], values)
    (root / 'aligned').mkdir(exist_ok=True)
    (root / 'clips').mkdir(exist_ok=True)
    timeline.to_csv(root / 'aligned' / 'pose_timeline.csv', index=False)
    clips = []
    for n, (slug, label, t) in enumerate(CLIPS, 1):
        clip_id = f'clip_{n:02d}_{slug}'
        clips.append({'id': clip_id, 'label': label, 'start_s': t,
                      'end_s': t + 8, 'duration_s': 8.0,
                      'source_host_start_s': t + START,
                      'ego_video': str(root / 'clips' / (clip_id + '_ego.mp4')),
                      'third_video': str(root / 'clips' / (clip_id + '_third_person.mp4'))})
    archive = root.parent / (EPISODE + '.zip')
    result = {
        'schema_version': 1, 'source_archive': str(archive),
        'source_archive_sha256': sha256(archive), 'source_directory': str(source),
        'source_pose_csv': str(source / 'aligned_pose.csv'),
        'source_frames_jsonl': str(source / 'frames_raw.jsonl'),
        'source_first_timeline_ns': int(poses.timeline_ns.iloc[0]),
        'source_ego_fps': 15.0, 'aligned_start_host_s': START,
        'aligned_end_host_s': END, 'aligned_duration_s': END - START,
        'fps': FPS, 'output_frames': len(output_t),
        'ego_video': str(root / 'aligned' / 'ego.mp4'),
        'third_video': str(root / 'aligned' / 'third_person.mp4'),
        'pose_timeline_csv': str(root / 'aligned' / 'pose_timeline.csv'),
        'alignment': {
            'time_origin': 'First ego frame timeline_ns (host receive clock).',
            'ego_mapping': 'Nearest original frame on exact timeline_ns; resampled at 15 Hz on host time.',
            'third_mapping': 'third_source_s = aligned_time_s + aligned_start_host_s + third_offset_from_host_s',
            'third_offset_from_host_s': THIRD_OFFSET,
            'visual_alignment_uncertainty_s': 0.2,
            'uncertainty_kind': 'Approximate visual review tolerance; not a statistical or physical synchronization bound.',
            'ego_playback_rate_correction': original_manifest['stream_quality']['timing']['host_seconds_per_media_second'],
            'physical_sync_verified': False,
            'events': [
                {'event': 'initial departure', 'ego_source_playback_s': 1.65,
                 'ego_host_s': 1.73, 'third_source_s': 4.85, 'estimated_offset_s': 3.12},
                {'event': 'final stop after returning to box', 'ego_source_playback_s': 164.4,
                 'ego_host_s': 172.67, 'third_source_s': 175.9, 'estimated_offset_s': 3.23},
            ],
            'review': 'Initial departure and final stop reviewed in 0.1 s contact sheets; route, doorway and turn sequence checked throughout at 8 s intervals.',
            'maximum_ego_frame_rounding_s': float(np.max(np.abs(host[ix] - target_host))),
        },
        'trim': {
            'reason': 'Remove long stationary ego tail after final arrival; retain 0.5 s settling after stop.',
            'ego_source_start_s': float(np.interp(START, host, poses.frame_index / 15)),
            'ego_source_end_s': float(np.interp(END, host, poses.frame_index / 15)),
            'third_source_start_s': START + THIRD_OFFSET,
            'third_source_end_s': END + THIRD_OFFSET,
            'removed_initial_host_s': START,
            'removed_terminal_host_s': float(host[-1] - END),
        },
        'ground_truth': {
            'kind': 'Recorded Go2 local odometry, not independent motion-capture ground truth.',
            'position_units': 'm', 'angle_units': 'rad',
            'invalid_source_frame_indices': poses.loc[~valid, 'frame_index'].astype(int).tolist(),
            'limitation': original_manifest['alignment']['limitation'],
        },
        'clip_selection': 'Representative route events selected before evaluating either model; all clips last 8 s.',
        'clips': clips,
    }
    (root / 'alignment_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result, timeline


def encode_ego(root, manifest, timeline):
    dest = Path(manifest['ego_video'])
    source = Path(manifest['source_directory'])
    cap = cv2.VideoCapture(str(source / 'video.mp4'))
    width, height = int(cap.get(3)), int(cap.get(4))
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(FPS), '-i', '-',
            '-an', '-c:v', 'libx264', '-threads', '6', '-preset', 'fast', '-crf', '17',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)]
    print('COMMAND:', ' '.join(args), flush=True)
    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    current, frame = -1, None
    try:
        for n, target in enumerate(timeline.source_frame_index):
            while current < target:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Could not decode ego source frame {target}')
                current += 1
            proc.stdin.write(frame.tobytes())
            if n % 450 == 0:
                print(f'Ego frames {n}/{len(timeline)}', flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('Ego encoder failed')
    finally:
        cap.release()
        if proc.poll() is None:
            proc.terminate()
    print(f'WROTE {dest}', flush=True)


def encode_third(root, manifest):
    # Decode HLG/BT.2020 to linear light, convert gamut, tone-map and encode SDR.
    # The nwm environment's ffmpeg lacks zscale; use the existing static build.
    source = Path(manifest['source_directory']) / 'video_3.mp4'
    command([HDR_FFMPEG, '-hide_banner', '-loglevel', 'warning', '-y',
             '-ss', START + THIRD_OFFSET, '-i', source,
             '-t', END - START, '-map', '0:v:0', '-an',
             '-vf', 'fps=15,zscale=w=1280:h=720:t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p',
             '-c:v', 'libx264', '-threads', '6', '-preset', 'fast', '-crf', '17',
             '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
             '-movflags', '+faststart', manifest['third_video']])


def clip_videos(manifest):
    for clip in manifest['clips']:
        for source_key, dest_key in [('ego_video', 'ego_video'), ('third_video', 'third_video')]:
            command(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
                     '-ss', clip['start_s'], '-i', manifest[source_key], '-t', clip['duration_s'],
                     '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '18',
                     '-pix_fmt', 'yuv420p', '-movflags', '+faststart', clip[dest_key]])


def contact_sheets(root, manifest):
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 21)
    a, b = [cv2.VideoCapture(manifest[k]) for k in ['ego_video', 'third_video']]
    for clip in manifest['clips']:
        sheet = Image.new('RGB', (1280, 4 * 214 + 65), '#101725')
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 18), f"{clip['label']}  |  matched views  |  8 seconds", fill='white', font=font)
        for n, rel in enumerate([0, 2, 4, 6]):
            t = clip['start_s'] + rel
            y = 65 + n * 214
            for col, (cap, name) in enumerate([(a, 'EGO'), (b, 'THIRD PERSON')]):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, fr = cap.read()
                if ok:
                    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((320, 180))
                    x = col * 640 + 280
                    sheet.paste(im, (x, y + 20))
                draw.text((col * 640 + 20, y + 75), f'{name}  {t:.1f}s', fill='#ced7e6', font=font)
        sheet.save(root / 'inspection' / (clip['id'] + '_aligned.jpg'), quality=93)
    a.release(); b.release()


def verify(manifest):
    paths = [manifest['ego_video'], manifest['third_video']]
    for clip in manifest['clips']:
        paths.extend([clip['ego_video'], clip['third_video']])
    result = []
    for name in paths:
        cap = cv2.VideoCapture(name)
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,width,height,avg_frame_rate,nb_frames,duration,pix_fmt',
            '-of', 'json', name], text=True))['streams'][0]
        expected = 2580 if name in paths[:2] else 120
        if fps != 15 or count != expected or int(probe['nb_frames']) != expected:
            raise RuntimeError(f'Unexpected video timing {name}: {fps} fps, {count} frames')
        result.append({'path': name, 'fps': fps, 'frames': count, 'duration_s': count / fps,
                       'size_bytes': Path(name).stat().st_size, 'ffprobe': probe})
    Path(manifest['ego_video']).parent.parent.joinpath('video_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('VALIDATED', len(result), 'videos: exact matching frame counts and durations.', flush=True)
    return result


def processing_report(root, manifest, validation):
    original_hash = sha256(Path(manifest['source_archive']))
    if original_hash != manifest['source_archive_sha256']:
        raise RuntimeError('Source archive checksum changed')
    third_probe = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,pix_fmt,color_primaries,color_transfer,color_space,nb_frames,duration',
        '-of', 'json', manifest['third_video']], text=True)
    third_probe = json.loads(third_probe)['streams'][0]
    assert third_probe['color_primaries'] == 'bt709'
    assert third_probe['color_transfer'] == 'bt709'
    assert third_probe['color_space'] == 'bt709'
    report = {
        'alignment_manifest_sha256': sha256(root / 'alignment_manifest.json'),
        'source_archive_sha256': original_hash, 'source_archive_unchanged': True,
        'third_person_color_conversion': {
            'source': '10-bit HLG / BT.2020 phone video',
            'output': '8-bit SDR / BT.709',
            'ffmpeg_binary': HDR_FFMPEG,
            'filter': 'fps=15,zscale=w=1280:h=720:t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p',
            'probe': third_probe,
        },
        'videos': validation,
        'visual_review': {
            'sheet': str(root / 'inspection' / 'aligned_review.jpg'),
            'aligned_times_s': [20, 60, 115, 170],
            'result': 'Corresponding doorway, left turn, glass doorway and final approach visually agree.',
        },
        'validation_exit_status': 0,
    }
    (root / 'processing_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Source archive integrity and third-person SDR metadata verified.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--skip-full', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        manifest = json.loads((args.root / 'alignment_manifest.json').read_text())
        processing_report(args.root, manifest, verify(manifest))
        return
    manifest, timeline = write_manifest(args.root)
    print('Manifest and pose timeline ready.', flush=True)
    if args.metadata_only:
        return
    if not args.skip_full:
        encode_ego(args.root, manifest, timeline)
        encode_third(args.root, manifest)
    clip_videos(manifest)
    contact_sheets(args.root, manifest)
    processing_report(args.root, manifest, verify(manifest))


if __name__ == '__main__':
    main()

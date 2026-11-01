#!/usr/bin/env python3
"""Preserve Go2 recording ZIPs and produce clock-labelled visual inventories.

Only decoders inspect media, including videos with unusual suffixes. This tool
never loads tensor/pickle files and never assigns a cross-camera alignment.
Large original files and review assets stay under the explicitly selected NAS
root. Existing files are verified against their ZIP CRC before reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import stat
import subprocess
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFAULT_NAS = Path('/file_system/nas/algorithm/dujun.nie')
DEFAULT_ROOT = DEFAULT_NAS / 'opennwm_go2_additions_20261030'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def hash_file(path):
    sha = hashlib.sha256()
    crc = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
            crc = zlib.crc32(block, crc)
    return sha.hexdigest(), crc & 0xffffffff


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))


def media_summary(data):
    stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
    return {**{k: stream.get(k) for k in (
        'codec_name', 'width', 'height', 'pix_fmt', 'r_frame_rate', 'avg_frame_rate',
        'nb_frames', 'color_space', 'color_transfer', 'color_primaries', 'color_range')},
        'duration_s': float(stream.get('duration', data['format']['duration'])),
        'format_name': data['format']['format_name'],
        'audio_streams': sum(s['codec_type'] == 'audio' for s in data['streams']),
        'tags': data['format'].get('tags', {}),
        'stream_tags': stream.get('tags', {})}


def extract(archive, root):
    episode = archive.stem
    source = root / 'sources' / episode
    receipt = root / 'audit' / 'source_inventory' / (episode + '.json')
    if receipt.exists():
        inventory = json.loads(receipt.read_text())
        if inventory.get('extraction_complete') and inventory.get('source_archive_size_bytes') == archive.stat().st_size:
            print('EXISTING_INVENTORY', receipt, flush=True)
            return inventory
    source.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        required_bytes = sum(m.file_size for m in members)
        free = shutil.disk_usage(root).free
        if free < required_bytes + 1024 ** 3:
            raise OSError(f'Insufficient disk: need {required_bytes + 1024 ** 3}, have {free}')
        members.sort(key=lambda m: ('/frames/' in m.filename, m.filename))
        seen = set()
        for index, member in enumerate(members):
            rel = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != episode
                    or stat.S_ISLNK(mode) or '\\' in member.filename):
                raise ValueError(f'Unsafe/unexpected ZIP member: {member.filename}')
            local = Path(*rel.parts[1:])
            if not local.parts or local.as_posix() in seen:
                raise ValueError(f'Duplicate/empty member: {member.filename}')
            seen.add(local.as_posix())
            dest = source / local
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.resolve().is_relative_to(source.resolve()):
                raise ValueError(f'Unsafe destination: {dest}')
            digest = None
            if dest.is_file() and dest.stat().st_size == member.file_size:
                digest, crc = hash_file(dest)
                if crc != member.CRC:
                    digest = None
            if digest is None:
                temp = dest.with_name(dest.name + '.extracting')
                sha = hashlib.sha256()
                with z.open(member) as a, temp.open('wb') as b:
                    for block in iter(lambda: a.read(8 * 1024 * 1024), b''):
                        b.write(block)
                        sha.update(block)
                if temp.stat().st_size != member.file_size:
                    raise IOError(f'Extracted size mismatch: {member.filename}')
                temp.replace(dest)
                digest = sha.hexdigest()
            files.append({'path': local.as_posix(), 'size_bytes': member.file_size,
                          'zip_crc32': f'{member.CRC:08x}', 'sha256': digest})
            if index % 1000 == 0 or '/frames/' not in member.filename:
                print('EXTRACT', episode, index + 1, len(members), local, flush=True)
    return source_inventory(archive, root, files, True)


def source_inventory(archive, root, files, complete):
    episode = archive.stem
    source = root / 'sources' / episode
    suffix = '.json' if complete else '.early.json'
    receipt = root / 'audit' / 'source_inventory' / (episode + suffix)
    third_candidates = sorted(p for p in source.glob('video_3.*') if not p.name.endswith('extracting'))
    if len(third_candidates) != 1:
        raise ValueError(f'Expected exactly one third-person media file: {third_candidates}')
    third_path = third_candidates[0]
    with third_path.open('rb') as handle:
        magic = handle.read(32).hex()
    ego_probe, third_probe = probe(source / 'video.mp4'), probe(third_path)
    original = json.loads((source / 'manifest.json').read_text())
    poses = pd.read_csv(source / 'aligned_pose.csv')
    timing = original['stream_quality']['timing']
    inventory = {
        'schema_version': 1, 'episode_id': episode,
        'source_archive': str(archive), 'source_archive_size_bytes': archive.stat().st_size,
        'source_archive_sha256': None, 'source_directory': str(source),
        'extraction_complete': complete, 'archive_members': len(files),
        'uncompressed_bytes': sum(f['size_bytes'] for f in files), 'files': files,
        'ego_video': str(source / 'video.mp4'), 'third_video': str(third_path),
        'third_magic_first_32_bytes_hex': magic,
        'ego_probe': ego_probe, 'third_probe': third_probe,
        'ego': media_summary(ego_probe), 'third': media_summary(third_probe),
        'pose_columns': list(poses.columns), 'pose_rows': len(poses),
        'first_timeline_ns': int(poses.timeline_ns.iloc[0]),
        'last_timeline_ns': int(poses.timeline_ns.iloc[-1]),
        'host_span_s': (int(poses.timeline_ns.iloc[-1]) - int(poses.timeline_ns.iloc[0])) / 1e9,
        'host_seconds_per_media_second': timing['host_seconds_per_media_second'],
        'source_timing': timing, 'source_recording_status': original['status'],
        'source_errors': original.get('errors', []), 'source_warnings': original.get('warnings', []),
        'invalid_pose_indices': original['pairing']['invalid_frame_indices'],
        'source_alignment': original['alignment'], 'cross_camera_alignment': None,
        'source_start_utc': original.get('start_utc'), 'sheets': []}
    write_json(receipt, inventory)
    print('SOURCE_READY', episode, json.dumps({k: inventory[k] for k in ['host_span_s', 'ego', 'third', 'third_video']}), flush=True)
    return inventory


def sheet(images, labels, dest, title, cols=5, per_page=30):
    font = ImageFont.truetype(FONT, 17)
    width, height, label_height = 320, 180, 27
    pages = []
    for start in range(0, len(images), per_page):
        group = list(zip(images[start:start + per_page], labels[start:start + per_page]))
        page = start // per_page + 1
        canvas = Image.new('RGB', (cols * width, 42 + math.ceil(len(group) / cols) * (height + label_height)), '#11191e')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 9), f'{title} | page {page}', font=font, fill='white')
        for i, (frame, label) in enumerate(group):
            x, y = (i % cols) * width, 42 + (i // cols) * (height + label_height)
            draw.text((x + 6, y + 4), label, font=font, fill='white')
            canvas.paste(ImageOps.pad(frame, (width, height), color='black'), (x, y + label_height))
        path = dest.with_name(dest.stem + f'_{page:02d}.jpg')
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
        pages.append(str(path))
        print('SHEET_READY', path, flush=True)
    return pages


def review(inventory, root, interval, ffmpeg, from_video=False):
    import cv2
    episode = inventory['episode_id']
    folder = root / 'audit' / 'source_inventory' / (episode + ('_early' if from_video else ''))
    source = Path(inventory['source_directory'])
    poses = pd.read_csv(source / 'aligned_pose.csv')
    host = (poses.timeline_ns.to_numpy() - poses.timeline_ns.iloc[0]) / 1e9
    host_samples = np.arange(0, host[-1] - 0.01, interval)
    ego_images, ego_labels, ego_samples = [], [], []
    capture = cv2.VideoCapture(inventory['ego_video']) if from_video else None
    for t in host_samples:
        index = int(np.argmin(np.abs(host - t)))
        row = poses.iloc[index]
        if capture is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Failed ego sample at frame {row.frame_index}')
            ego_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            ego_images.append(Image.open(source / row['file']).convert('RGB'))
        ego_labels.append(f'host {host[index]:.2f}s | #{int(row.frame_index)}')
        ego_samples.append({'host_s': float(host[index]), 'frame_index': int(row.frame_index),
                            'video_s': float(row.frame_index) / float(Fraction(inventory['ego']['avg_frame_rate'])),
                            'frame': str(source / row['file'])})
    if capture is not None:
        capture.release()
    pages = sheet(ego_images, ego_labels, folder / 'ego_overview.jpg', episode + ' EGO HOST seconds')
    stream = inventory['third']
    tone_map = stream['color_transfer'] in ('arib-std-b67', 'smpte2084')
    if tone_map:
        filt = 'zscale=w=640:h=-2:t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=rgb24'
    else:
        filt = 'scale=640:-2,format=rgb24'
    third_images, third_labels, third_samples = [], [], []
    for t in np.arange(0, stream['duration_s'] - 0.01, interval):
        output = subprocess.check_output([ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', str(t),
            '-threads', '2', '-i', inventory['third_video'], '-frames:v', '1', '-an', '-vf', filt,
            '-threads', '2', '-filter_threads', '1', '-f', 'image2pipe', '-c:v', 'png', '-'])
        third_images.append(Image.open(io.BytesIO(output)).convert('RGB'))
        third_labels.append(f'source {t:.2f} s')
        third_samples.append(float(t))
    pages += sheet(third_images, third_labels, folder / 'third_overview.jpg', episode + ' THIRD source seconds')
    inventory['sheets'] = pages
    inventory['review'] = {'interval_s': interval, 'ego_samples': ego_samples,
                           'third_samples_source_s': third_samples, 'third_hdr_tonemapped_for_review': tone_map,
                           'note': 'Contact sheets show independent clocks; no cross-camera alignment is inferred.'}
    suffix = '.early.json' if from_video else '.json'
    write_json(root / 'audit' / 'source_inventory' / (episode + suffix), inventory)


def media_review_first(archives, root, interval, ffmpeg):
    """Allow early alignment review while a full extraction runs independently."""
    inventories = []
    for archive in archives:
        source = root / 'sources' / archive.stem
        source.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                rel = PurePosixPath(item.filename)
                if (len(rel.parts) != 2 or rel.parts[0] != archive.stem or item.is_dir()
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    continue
                if rel.name not in ('manifest.json', 'aligned_pose.csv', 'frames_raw.jsonl', 'video.mp4') and not rel.name.startswith('video_3.'):
                    continue
                dest = source / rel.name
                if not dest.exists() or dest.stat().st_size != item.file_size:
                    # A unique suffix safely coexists with full extraction.
                    temp = dest.with_name(dest.name + '.early-extracting')
                    with z.open(item) as a, temp.open('wb') as b:
                        shutil.copyfileobj(a, b, 8 * 1024 * 1024)
                    temp.replace(dest)
                    print('EARLY_MEDIA', archive.stem, dest.name, flush=True)
        inv = source_inventory(archive, root, [], False)
        review(inv, root, interval, ffmpeg, from_video=True)
        inventories.append(inv)
    write_json(root / 'audit' / 'source_inventory' / 'early_summary.json', inventories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--archive-glob', default=str(DEFAULT_NAS / 'episode_20261029*.zip'))
    parser.add_argument('--interval', type=float, default=12)
    parser.add_argument('--ffmpeg', default='/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/fastwam/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
    parser.add_argument('--skip-review', action='store_true')
    parser.add_argument('--skip-archive-hash', action='store_true')
    parser.add_argument('--media-review-only', action='store_true', help='Copy only media/metadata and create early video-based sheets.')
    parser.add_argument('--workers', type=int, default=1, help='Parallel archive extraction workers.')
    args = parser.parse_args()
    if not 0 < args.interval:
        parser.error('--interval must be positive')
    from glob import glob
    archives = sorted(Path(p) for p in glob(args.archive_glob))
    if not archives:
        parser.error('No matching archives')
    args.root.mkdir(parents=True, exist_ok=True)
    if args.media_review_only:
        media_review_first(archives, args.root, args.interval, args.ffmpeg)
        print('EARLY_REVIEW_COMPLETE', len(archives), flush=True)
        return
    def prepare_archive(archive):
        inv = extract(archive, args.root)
        if not args.skip_review and not inv.get('sheets'):
            early_path = args.root / 'audit' / 'source_inventory' / (archive.stem + '.early.json')
            early = json.loads(early_path.read_text()) if early_path.exists() else {}
            if early.get('sheets') and all(Path(p).is_file() for p in early['sheets']):
                inv['sheets'] = early['sheets']
                inv['review'] = early['review']
                write_json(args.root / 'audit' / 'source_inventory' / (archive.stem + '.json'), inv)
            else:
                review(inv, args.root, args.interval, args.ffmpeg)
        return inv
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        inventories = list(pool.map(prepare_archive, archives))
    # Defer large archive hashing until every episode is ready for visual review.
    if not args.skip_archive_hash:
        for inv in inventories:
            if not inv.get('source_archive_sha256'):
                inv['source_archive_sha256'] = hash_file(Path(inv['source_archive']))[0]
                write_json(args.root / 'audit' / 'source_inventory' / (inv['episode_id'] + '.json'), inv)
                print('ARCHIVE_HASH_READY', inv['episode_id'], inv['source_archive_sha256'], flush=True)
    compact = [{k: inv[k] for k in ['episode_id', 'source_archive', 'source_archive_sha256', 'source_directory',
        'ego_video', 'third_video', 'host_span_s', 'host_seconds_per_media_second', 'ego', 'third', 'sheets',
        'invalid_pose_indices', 'source_start_utc']} for inv in inventories]
    write_json(args.root / 'audit' / 'source_inventory' / 'summary.json', compact)
    print('COMPLETE', len(inventories), args.root / 'audit' / 'source_inventory' / 'summary.json', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Prepare preserved Go2 sources and visually calibrated multi-trajectory media.

All large outputs live on NAS. Source archives are read-only. Selection metadata
is frozen before model evaluation; independent prepare/encode stages are resumable.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

NAS = Path('/file_system/nas/algorithm/dujun.nie')
ROOT = NAS / 'opennwm_go2_showcase_20261027'
EPISODES = ['episode_20261026T091121_148a01b5', 'episode_20261027T032914_26ec5b0a',
            'episode_20261027T034722_7ee6d875', 'episode_20261027T041128_88e3cb76']
FFMPEG = '/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/fastwam/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    print('COMMAND', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))


def load_inventory(root, episode):
    """Accept original episode receipts and the newer preserved-source audit."""
    paths = [root / 'episodes' / episode / 'source_inventory.json',
             root / 'audit' / 'source_inventory' / (episode + '.json')]
    for path in paths:
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(f'No source inventory for {episode}: {paths}')


def source_video(source, kind, inventory=None, manifest=None):
    """Resolve preserved media by metadata; decoders determine its real format.

    QuickTime and mislabeled files are passed directly to ffprobe/ffmpeg. No
    tensor/pickle deserialization or replacement of original media is needed.
    """
    explicit = (manifest or {}).get(f'source_{kind}_video') or (inventory or {}).get(f'{kind}_video')
    if explicit:
        path = Path(explicit)
        path = path if path.is_absolute() else source / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(source.glob('video_3.*')) if kind == 'third' else [source / 'video.mp4']
    candidates = [path for path in candidates if path.is_file() and not path.name.endswith('.extracting')]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one {kind} source video: {candidates}')
    return candidates[0]


def video_stream(data):
    return next(stream for stream in data['streams'] if stream['codec_type'] == 'video')


def stream_duration(data):
    stream = video_stream(data)
    return float(stream.get('duration') or data['format']['duration'])


def extract(root, episode, archive_dir=NAS):
    archive = archive_dir / (episode + '.zip')
    source = root / 'sources' / episode
    receipt = root / 'episodes' / episode / 'source_inventory.json'
    if receipt.exists():
        print('EXTRACTED_ALREADY', episode, flush=True)
        return
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        # Metadata and both videos first so visual review can start immediately.
        infos.sort(key=lambda i: ('/frames/' in i.filename, i.filename))
        for i in infos:
            rel = Path(i.filename)
            if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != episode:
                raise ValueError(f'Unsafe or unexpected ZIP member: {i.filename}')
            dest = source.joinpath(*rel.parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == i.file_size:
                continue
            temporary = dest.with_name(dest.name + '.extracting')
            with z.open(i) as a, temporary.open('wb') as b:
                shutil.copyfileobj(a, b, 8 * 1024 * 1024)
            temporary.rename(dest)
    original = json.loads((source / 'manifest.json').read_text())
    poses = pd.read_csv(source / 'aligned_pose.csv')
    ego, third = source_video(source, 'ego'), source_video(source, 'third')
    obj = {'episode_id': episode, 'source_archive': str(archive), 'source_archive_sha256': digest(archive),
           'source_directory': str(source), 'archive_members': len(infos),
           'ego_video': str(ego), 'third_video': str(third),
           'ego_probe': probe(ego), 'third_probe': probe(third),
           'host_span_s': (int(poses.timeline_ns.iloc[-1]) - int(poses.timeline_ns.iloc[0])) / 1e9,
           'host_seconds_per_media_second': original['stream_quality']['timing']['host_seconds_per_media_second'],
           'invalid_pose_indices': original['pairing']['invalid_frame_indices']}
    write_json(receipt, obj)
    print('SOURCE_READY', episode, json.dumps({k: obj[k] for k in ['source_directory', 'host_span_s', 'host_seconds_per_media_second', 'invalid_pose_indices']}), flush=True)


def color_filter(stream, width=None):
    resize = f'w={width}:h=-2:' if width else ''
    if stream.get('color_transfer') in ('arib-std-b67', 'smpte2084'):
        return f'zscale={resize}t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p'
    return (f'scale={width}:-2,' if width else '') + 'format=yuv420p'


def sheets(frames, times, dest, title, cols=5, per_page=50, width=320, height=180):
    font = ImageFont.truetype(FONT, 18)
    for page in range((len(times) + per_page - 1) // per_page):
        chosen = list(zip(frames, times))[page * per_page:(page + 1) * per_page]
        rows = (len(chosen) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * width, 42 + rows * (height + 27)), '#11191e')
        d = ImageDraw.Draw(sheet)
        d.text((12, 8), f'{title} | page {page+1}', font=font, fill='white')
        for j, (frame, time) in enumerate(chosen):
            x, y = j % cols * width, 42 + j // cols * (height + 27)
            sheet.paste(frame.resize((width, height)), (x, y + 27))
            d.text((x + 7, y + 3), f'{time:.2f} s', font=font, fill='white')
        path = dest.with_name(dest.stem + f'_{page+1:02d}.jpg')
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, quality=92)
        print('SHEET', path, flush=True)


def inspect(root, episode, interval):
    inventory = load_inventory(root, episode)
    source = Path(inventory['source_directory'])
    work = root / '_work' / episode
    work.mkdir(parents=True, exist_ok=True)
    third_path = source_video(source, 'third', inventory)
    third = video_stream(inventory['third_probe'])
    proxy = work / 'third_review.mp4'
    if not proxy.exists():
        run([FFMPEG, '-hide_banner', '-loglevel', 'warning', '-y', '-i', third_path, '-an',
             '-vf', 'fps=4,' + color_filter(third, 640), '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '4', proxy])
    cap = cv2.VideoCapture(str(proxy))
    times = np.arange(0, stream_duration(inventory['third_probe']) - .01, interval)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t)*1000)
        ok, fr = cap.read()
        if not ok: break
        frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    sheets(frames, times[:len(frames)], work / 'third_overview.jpg', episode + ' THIRD source seconds')
    poses = pd.read_csv(source / 'aligned_pose.csv')
    host = (poses.timeline_ns.to_numpy() - poses.timeline_ns.iloc[0]) / 1e9
    ego_times = np.arange(0, host[-1] - .01, interval)
    frames = []
    for t in ego_times:
        ix = int(np.argmin(np.abs(host-t)))
        path = source / str(poses.iloc[ix]['file']) if 'file' in poses else source / 'frames' / f'{ix:07d}.jpg'
        if not path.exists():
            path = source / 'frames' / f'frame_{ix:06d}.jpg'
        if not path.exists():
            # Use the explicit recorder frame path, whose naming differs by version.
            raw = [json.loads(s) for s in (source/'frames_raw.jsonl').read_text().splitlines()]
            print('FRAME_KEYS', raw[0], flush=True)
            raise FileNotFoundError(path)
        frames.append(Image.open(path).convert('RGB'))
    sheets(frames, ego_times, work / 'ego_overview.jpg', episode + ' EGO host seconds')


def freeze(root, episode, selection, revise=False, fps=15):
    """Write the exact timing/selection contract once, before inference starts."""
    out = root / 'episodes' / episode
    target = out / 'alignment_manifest.json'
    if target.exists() and not revise:
        raise FileExistsError(f'Frozen manifest exists: {target}')
    if target.exists():
        history=out/'metadata'
        history.mkdir(exist_ok=True)
        shutil.copy2(target,history/('alignment_manifest.'+digest(target)[:12]+'.json'))
    inv = load_inventory(root, episode)
    source = Path(inv['source_directory'])
    ego_source, third_source = (source_video(source, kind, inv) for kind in ('ego', 'third'))
    source_fps = float(Fraction(video_stream(inv['ego_probe'])['avg_frame_rate']))
    original = json.loads((source / 'manifest.json').read_text())
    poses = pd.read_csv(source / 'aligned_pose.csv')
    host = (poses.timeline_ns.to_numpy() - poses.timeline_ns.iloc[0]) / 1e9
    start, end = selection['start_host_s'], selection['end_host_s']
    slope, offset = selection.get('third_clock_scale', 1.0), selection['third_offset_s']
    fps = int(selection.get('fps', fps))
    if fps < 1 or source_fps <= 0 or not 0 <= start < end <= host[-1] + .001:
        raise ValueError('Invalid frame rate or ego alignment range')
    if slope <= 0 or slope * start + offset < 0 or slope * end + offset > stream_duration(inv['third_probe']) + .001:
        raise ValueError('Aligned interval is outside third-person source media')
    if not inv.get('source_archive_sha256'):
        inv['source_archive_sha256'] = digest(Path(inv['source_archive']))
    # Freeze a self-contained inventory without modifying the ingestion audit.
    write_json(out / 'source_inventory.json', inv)
    aligned = np.arange(round((end-start)*fps)) / fps
    targets = aligned + start
    right = np.searchsorted(host, targets).clip(1, len(host)-1)
    ix = np.where(abs(host[right-1]-targets) < abs(host[right]-targets), right-1, right)
    timeline = pd.DataFrame({'output_frame_index': np.arange(len(aligned)), 'aligned_time_s': aligned,
        'source_host_s': targets, 'source_frame_index': poses.frame_index.to_numpy()[ix],
        'source_video_s': poses.frame_index.to_numpy()[ix]/source_fps, 'source_frame_host_s': host[ix],
        'nearest_frame_time_error_s': host[ix]-targets, 'third_source_s': slope*targets+offset,
        'pair_valid': poses.pair_valid.to_numpy()[ix]})
    valid = poses.pair_valid.fillna(False).astype(bool).to_numpy()
    for key in ['x','y','z','yaw','vx','vy','yaw_speed']:
        values = poses[key].to_numpy()[valid]
        timeline[key] = np.interp(targets, host[valid], np.unwrap(values) if key=='yaw' else values)
    (out/'aligned').mkdir(parents=True, exist_ok=True)
    (out/'clips').mkdir(exist_ok=True)
    timeline.to_csv(out/'aligned/pose_timeline.csv', index=False)
    clips=[]
    for n, item in enumerate(selection['clips'], 1):
        item=dict(item)
        item['id']=f'clip_{n:02d}_'+item.pop('slug')
        item['duration_s']=item['end_s']-item['start_s']
        assert 20 <= item['duration_s'] <= 30
        assert 0 <= item['start_s'] < item['end_s'] <= end-start
        item['source_host_start_s']=start+item['start_s']
        item['ego_video']=str(out/'clips'/(item['id']+'_ego.mp4'))
        item['third_video']=str(out/'clips'/(item['id']+'_third_person.mp4'))
        clips.append(item)
    result={'schema_version':2, 'episode_id':episode,
        **{k:inv[k] for k in ['source_archive','source_archive_sha256','source_directory']},
        'source_ego_video':str(ego_source),'source_third_video':str(third_source),
        'source_pose_csv':str(source/'aligned_pose.csv'), 'source_frames_jsonl':str(source/'frames_raw.jsonl'),
        'source_first_timeline_ns':int(poses.timeline_ns.iloc[0]),'source_ego_fps':source_fps,
        'aligned_start_host_s':start,'aligned_end_host_s':end,'aligned_duration_s':end-start,
        'fps':fps,'output_frames':len(aligned),'ego_video':str(out/'aligned/ego.mp4'),
        'third_video':str(out/'aligned/third_person.mp4'),'pose_timeline_csv':str(out/'aligned/pose_timeline.csv'),
        'alignment':{'time_origin':'First ego timeline_ns; host receive clock.',
          'ego_mapping':f'Nearest actual timeline_ns source frame, resampled on host time at {fps} Hz; repeated frames are not interpolated.',
          'third_mapping':'third_source_s = third_clock_scale * source_host_s + third_offset_from_host_s',
          'third_clock_scale':slope,'third_offset_from_host_s':offset,
          'ego_playback_rate_correction':inv['host_seconds_per_media_second'],
          'visual_alignment_uncertainty_s':selection.get('uncertainty_s',.25),
          'uncertainty_kind':'Approximate visual review tolerance, not a physical/statistical synchronization bound.',
          'physical_sync_verified':False,'events':selection['events'],
          'maximum_ego_frame_rounding_s':float(np.max(abs(host[ix]-targets)))},
        'trim':{'reason':selection['trim_reason'],'third_source_start_s':slope*start+offset,
          'third_source_end_s':slope*end+offset,'removed_initial_host_s':start,'removed_terminal_host_s':float(host[-1]-end)},
        'ground_truth':{'kind':'Recorded Go2 local odometry, not independent motion capture.',
          'position_units':'m','angle_units':'rad','invalid_source_frame_indices':inv['invalid_pose_indices'],
          'limitation':original['alignment']['limitation']},
        'clip_selection':{'method':'Scene/motion diversity and third-person visibility selected before model evaluation.',
          'review_hz':selection.get('review_hz'), 'review':selection['review'],'excluded_third_source_ranges_s':selection.get('excluded_third_source_ranges_s',[])},
        'clips':clips}
    write_json(target,result)
    print('MANIFEST_FROZEN',target,'sha256',digest(target),flush=True)


def encode(root, episode):
    out=root/'episodes'/episode
    manifest=json.loads((out/'alignment_manifest.json').read_text())
    inv=load_inventory(root, episode)
    source=Path(manifest['source_directory'])
    fps=manifest['fps']
    ego=Path(manifest['ego_video'])
    if not ego.exists():
        timeline=pd.read_csv(manifest['pose_timeline_csv'])
        cap=cv2.VideoCapture(str(source_video(source, 'ego', inv, manifest)))
        w,h=int(cap.get(3)),int(cap.get(4))
        temp=ego.with_name('ego.partial.mp4')
        args=[FFMPEG,'-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','bgr24',
            '-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4',
            '-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
        print('COMMAND',' '.join(args),flush=True)
        p=subprocess.Popen(args,stdin=subprocess.PIPE)
        current,frame=-1,None
        try:
            for n,target in enumerate(timeline.source_frame_index):
                while current<target:
                    ok,frame=cap.read()
                    if not ok: raise RuntimeError(f'Missing ego source frame {target}')
                    current+=1
                p.stdin.write(frame.tobytes())
                if n%900==0: print('EGO_PROGRESS',episode,n,len(timeline),flush=True)
            p.stdin.close()
            if p.wait()!=0: raise RuntimeError('Ego ffmpeg failed')
            temp.rename(ego)
        finally:
            cap.release()
            if p.poll() is None: p.terminate()
    third=Path(manifest['third_video'])
    if not third.exists():
        stream=video_stream(inv['third_probe'])
        scale=manifest['alignment']['third_clock_scale']
        temp=third.with_name('third_person.partial.mp4')
        vf=f'setpts=(PTS-STARTPTS)/{scale},fps={fps},'+color_filter(stream)
        run([FFMPEG,'-hide_banner','-loglevel','warning','-y','-ss',manifest['trim']['third_source_start_s'],
            '-i',source_video(source, 'third', inv, manifest),'-an','-vf',vf,'-frames:v',manifest['output_frames'],
            '-c:v','libx264','-threads','4','-preset','fast','-crf','17','-pix_fmt','yuv420p',
            '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',temp])
        temp.rename(third)
    for clip in manifest['clips']:
        for key in ['ego_video','third_video']:
            dest=Path(clip[key])
            if dest.exists(): continue
            temp=dest.with_name(dest.stem+'.partial.mp4')
            run([FFMPEG,'-hide_banner','-loglevel','warning','-y','-ss',clip['start_s'],'-i',manifest[key],
                '-an','-frames:v',round(clip['duration_s']*fps),'-c:v','libx264','-threads','4','-preset','fast',
                '-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',temp])
            temp.rename(dest)
    validate(root,episode)


def validate(root,episode):
    out=root/'episodes'/episode
    m=json.loads((out/'alignment_manifest.json').read_text())
    records=[]
    for obj,duration in [(m,m['aligned_duration_s'])]+[(c,c['duration_s']) for c in m['clips']]:
        for key in ['ego_video','third_video']:
            path=Path(obj[key]); info=probe(path)
            s=next(s for s in info['streams'] if s['codec_type']=='video')
            assert not any(stream['codec_type']=='audio' for stream in info['streams']), f'Audio stream present: {path}'
            assert int(s['nb_frames'])==round(duration*m['fps']), (path,s)
            assert abs(float(Fraction(s['avg_frame_rate']))-m['fps'])<.001,(path,s)
            assert abs(float(s['duration'])-duration)<.01,(path,s)
            records.append({'path':str(path),'duration_s':float(s['duration']),'frames':int(s['nb_frames']),
                            'width':s['width'],'height':s['height'],'sha256':digest(path)})
    checksum=digest(Path(m['source_archive']))
    assert checksum==m['source_archive_sha256'],'Source archive changed'
    write_json(out/'media_validation.json',{'manifest_sha256':digest(out/'alignment_manifest.json'),
        'original_archive_unchanged':True,'original_archive_sha256':checksum,'videos':records,
        'color_conversion':'HLG/PQ sources linearized and Hable tone mapped to SDR BT.709 before encoding.',
        'complete':True})
    print('EPISODE_COMPLETE',episode,flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--archive-dir', type=Path, default=NAS)
    ap.add_argument('--episode', action='append')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--interval', type=float, default=8)
    ap.add_argument('--selection', type=Path)
    ap.add_argument('--fps', type=int, default=15,
                    help='Shared aligned camera frame rate; 30 preserves 30 Hz third-person motion while repeating ego frames.')
    ap.add_argument('--revise-selection', action='store_true')
    ap.add_argument('--encode', action='store_true')
    ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()
    for episode in args.episode or EPISODES:
        if args.extract: extract(args.root, episode, args.archive_dir)
        if args.inspect: inspect(args.root, episode, args.interval)
        if args.selection: freeze(args.root, episode, json.loads(args.selection.read_text())[episode], args.revise_selection, args.fps)
        if args.encode: encode(args.root, episode)
        if args.validate: validate(args.root, episode)


if __name__ == '__main__':
    main()

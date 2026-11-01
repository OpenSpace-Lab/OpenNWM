#!/usr/bin/env python3
"""Build first-person website films from locally curated navigation frames.

Run in the base conda environment (OpenCV, NumPy, Pillow, system ffmpeg).
Intermediate caches stay on NAS; compressed exports enter the site.
The public cinematic manifest is also a portable fallback curation source.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
NAV = Path(os.environ.get('OPENNWM_NAV_ROOT', '/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere'))
LUSNAR = Path(os.environ.get('OPENNWM_LUSNAR_ROOT', '/file_system/nas/algorithm/dujun.nie/datasets/LuSNAR'))
PLANETARY = Path(os.environ.get('OPENNWM_PLANETARY_ROOT', '/file_system/nas/algorithm/dujun.nie/nwm/data/planetary_rover'))
SYNMARS = Path(os.environ.get('OPENNWM_SYNMARS_ROOT', '/file_system/nas/algorithm/dujun.nie/datasets/SynMars-TW'))
LEGACY = Path('/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/redesign_20260929/cinematic')
WORK = Path(os.environ.get('OPENNWM_MEDIA_WORK_DIR', str(LEGACY.parent/'cinematic_v3')))
OUT = REPO/'website/assets/hero'
MANIFEST = REPO/'website/content/cinematic.json'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FPS, SIZE, GRID = 24, (1920, 1080), 15
TILE_SIZE, MOSAIC_DURATION, CROSSFADE = (320, 180), 3., .18
cv2.setNumThreads(2)


@lru_cache(maxsize=256)
def frames(directory: Path) -> list[Path]:
    # The NAS can repeat an entire readdir page indefinitely. Stop after a
    # duplicate-only cycle; retaining a set also handles finite repeated pages.
    seen, duplicate_run = set(), 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name in seen:
                duplicate_run += 1
                if duplicate_run >= max(256, len(seen)):
                    break
            else:
                seen.add(entry.name)
                duplicate_run = 0
    return sorted({directory/name for name in seen
                   if Path(name).suffix.lower() in {'.jpg', '.jpeg', '.png'}},
                  key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem))


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    ih, iw = image.shape[:2]
    scale = max(width/iw, height/ih)
    rw, rh = max(width, round(iw*scale)), max(height, round(ih*scale))
    image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return image[(rh-height)//2:(rh-height)//2+height, (rw-width)//2:(rw-width)//2+width]


def read(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return cover(image, width, height)


def curation() -> dict:
    path = WORK/'curation.json'
    if path.exists():
        return json.loads(path.read_text())
    doc = json.loads(MANIFEST.read_text())
    if doc.get('version', 0) < 2:
        raise ValueError(f'Curated scene labels and selections required: {path}')
    return {'wall': doc['clips'], 'hero': [r for r in doc['journeyScenes'] if r.get('planet') != 'wall']}


def normalize(raw: dict, lookup: dict | None = None) -> dict:
    record = {**(lookup or {}).get(raw['id'], {}), **raw}
    if 'first_frame' in raw and 'frame_paths' not in raw:
        record.pop('frame_paths', None)
        if 'frame_names' not in raw:
            record.pop('frame_names', None)
    source = record['source']
    collection = record.get('collection', 'LuSNAR' if source == 'LuSNAR' else
                            'Planetary Rover' if 'Zhurong' in source else 'NavAnywhere')
    record['collection'] = collection
    directory = record.get('directory') or record.get('source_path')
    selected_paths = record.get('frame_paths')
    if directory:
        directory = Path(directory)
        if directory.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            record['first_frame'] = directory.name
            directory = directory.parent
    elif selected_paths:
        first = Path(selected_paths[0])
        directory = (first if first.is_absolute() else NAV/first).parent
    elif collection == 'LuSNAR':
        directory = LUSNAR/record['trajectory']/'image0/color'
    elif collection == 'Planetary Rover':
        directory = PLANETARY/record['trajectory']
    elif collection == 'SynMars-TW':
        directory = SYNMARS/'images'/record.get('split', 'testing')
    else:
        directory = NAV/source/record['trajectory']
    record['directory'] = str(directory)
    all_items = None
    if record.get('frame_names'):
        paths = [directory/name for name in record['frame_names']]
    elif selected_paths:
        paths = [Path(p) if Path(p).is_absolute() else NAV/p for p in selected_paths]
    elif record.get('first_frame') and Path(record['first_frame']).stem.isdigit():
        # Earth trajectories have numbered contiguous frames. Use the curated
        # names rather than re-listing NAS directories, whose paged listings
        # can omit entries as well as repeating them.
        first = Path(record['first_frame'])
        start = int(first.stem)
        padding = len(first.stem) if first.stem.startswith('0') and len(first.stem) > 1 else 0
        paths = [directory/f'{index:0{padding}d}{first.suffix}'
                 for index in range(start, start+int(record['frame_count']))]
    else:
        all_items = frames(directory)
        names = [p.name for p in all_items]
        start = names.index(record['first_frame']) if record.get('first_frame') in names else record.get('first_index', 0)
        count = min(int(record.get('frame_count', 72)), len(all_items)-start)
        paths = all_items[start:start+count]
    if not paths or not all(p.is_file() for p in paths):
        raise ValueError(f'Missing selected source frames for {record["id"]}')
    kind = record.get('kind', 'video')
    if kind in {'image', 'mission_still'}:
        paths, kind = paths[:1], 'mission_still'
    else:
        kind = 'simulation_sequence' if collection in {'LuSNAR', 'SynMars-TW'} else 'dataset_sequence'
    record.update(kind=kind, frame_paths=[str(p) for p in paths], frame_names=[p.name for p in paths], first_frame=paths[0].name,
                  last_frame=paths[-1].name, frame_count=len(paths),
                  trajectory_frame_count=record.get('trajectory_frame_count', len(all_items) if all_items else len(paths)),
                  first_index=all_items.index(paths[0]) if all_items else record.get('first_index', 0))
    record['labelEn'] = record.get('labelEn') or record.get('scene_name')
    record['labelZh'] = record.get('labelZh') or record['labelEn']
    if not record['labelEn']:
        raise ValueError(f'Missing scene label for {record["id"]}')
    if collection == 'LuSNAR':
        record['relative_path'] = f'LuSNAR/{record["trajectory"]}/image0/color'
    elif collection == 'Planetary Rover':
        record['relative_path'] = f'planetary_rover/{record["trajectory"]}'
    elif collection == 'SynMars-TW':
        record['relative_path'] = f'SynMars-TW/images/{record.get("split", "testing")}'
    else:
        record['relative_path'] = f'NavAnywhere/{source}/{record["trajectory"]}'
    return record


def selections() -> tuple[list[dict], list[dict]]:
    doc = curation()
    wall = [normalize(raw) for raw in doc['wall']]
    if len(wall) != GRID*GRID or len({(r['source'], r['trajectory']) for r in wall}) != GRID*GRID:
        raise ValueError('The wall must contain 225 distinct source trajectories')
    if len({r['labelEn'] for r in wall}) != len(wall):
        raise ValueError('Every wall tile needs a unique short scene label')
    if len({r['source'] for r in wall}) != 15:
        raise ValueError('The wall must cover all 15 NavAnywhere sources')
    lookup = {r['id']: r for r in wall}
    hero = [normalize(raw, lookup) for raw in doc['hero'] if raw.get('planet') in {'moon', 'mars'}]
    if len({r['labelEn'] for r in hero}) != len(hero):
        raise ValueError('Every journey segment needs a unique short scene label')
    planets = [r['planet'] for r in hero]
    if list(dict.fromkeys(planets)) != ['moon', 'mars']:
        raise ValueError('Journey order must be Moon, then Mars')
    if any(r['source'] == 'CASIA-Nav' and r['trajectory'].startswith('office') for r in wall+hero):
        raise ValueError('CASIA office trajectories must not appear')
    return wall, hero


def sample() -> None:
    """Validate reviewed curation and export labeled contact sheets."""
    WORK.mkdir(parents=True, exist_ok=True)
    wall, hero = selections()
    font = ImageFont.truetype(FONT, 18)
    for name, records in [('wall', wall), ('hero', hero)]:
        columns = 5 if name == 'wall' else 4
        sheet = Image.new('RGB', (columns*320, math.ceil(len(records)/columns)*208), '#0a111d')
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(records):
            path = Path(record['frame_paths'][len(record['frame_paths'])//2])
            tile = Image.fromarray(cv2.cvtColor(read(path, *TILE_SIZE), cv2.COLOR_BGR2RGB))
            x, y = index%columns*320, index//columns*208
            sheet.paste(tile, (x, y))
            draw.text((x+6, y+184), record['labelEn'], font=font, fill='#f1f7fd')
        sheet.save(WORK/f'{name}-contact-sheet.jpg', quality=90)
    print(f'SAMPLE COMPLETE: {len(wall)} wall clips; {len(hero)} journey scenes', flush=True)


def cache_name(record: dict) -> str:
    signature = '\n'.join([record['source'], record['trajectory'], *record['frame_names']])
    suffix = hashlib.blake2b(signature.encode(), digest_size=6).hexdigest()
    return f'{record["id"]}-{suffix}.npy'


def prepare() -> None:
    """Reuse matching legacy tiles and cache only newly selected source frames."""
    WORK.mkdir(parents=True, exist_ok=True)
    wall, hero = selections()
    old_path = LEGACY/'selected.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else []
    legacy = {(r['source'], r['trajectory'], r['first_frame'], r['frame_count']): r for r in old}
    cache = WORK/'tiles'
    cache.mkdir(exist_ok=True)
    if all('grid_index' in record for record in wall):
        wall.sort(key=lambda record: record['grid_index'])
    else:
        np.random.default_rng(20260930).shuffle(wall)
    def cache_record(pair: tuple[int, dict]) -> None:
        index, record = pair
        key = (record['source'], record['trajectory'], record['first_frame'], record['frame_count'])
        prior = legacy.get(key)
        reuse = prior and Path(prior.get('cache', '')).is_file()
        existing = Path(record.get('cache', '/missing'))
        target = existing if existing.is_file() else Path(prior['cache']) if reuse else cache/cache_name(record)
        if not target.exists():
            data = np.stack([read(Path(p), *TILE_SIZE) for p in record['frame_paths']])
            np.save(target, data)
        data = np.load(target, mmap_mode='r')
        if data.shape != (record['frame_count'], TILE_SIZE[1], TILE_SIZE[0], 3):
            raise ValueError(f'Cache shape mismatch: {target}: {data.shape}')
        record.update(cache=str(target), grid_index=index,
                      playback_fps=round(min(12., max(2., record['frame_count']/6.)), 3))
        print(f'{"REUSED" if reuse else "CACHED"} {index+1}/{len(wall)} {record["id"]} | {record["labelEn"]}', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(cache_record, enumerate(wall)))
    (WORK/'selected.json').write_text(json.dumps(wall, indent=2)+'\n')
    (WORK/'hero-selected.json').write_text(json.dumps(hero, indent=2)+'\n')
    print('PREPARE COMPLETE', flush=True)


class Encoder:
    def __init__(self, path: Path, crf: int = 25):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temporary = path.with_name(f'.{path.stem}.encoding.mp4')
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS), '-i', '-', '-an',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-threads', '8',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.temporary)], stdin=subprocess.PIPE)
    def write(self, frame: np.ndarray) -> None:
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
    def close(self) -> None:
        self.process.stdin.close()
        result = self.process.wait()
        if result:
            raise RuntimeError(f'ffmpeg exited {result} encoding {self.path}')
        self.temporary.replace(self.path)


def label_strip(text: str) -> np.ndarray:
    image = Image.new('RGB', (316, 29), '#101824')
    draw = ImageDraw.Draw(image)
    size = 22
    font = ImageFont.truetype(FONT, size)
    while draw.textlength(text, font=font) > 300 and size > 12:
        size -= 1
        font = ImageFont.truetype(FONT, size)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((8, (29-(box[3]-box[1]))//2-box[1]), text, font=font, fill='#f1f7fd')
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class Mosaic:
    def __init__(self):
        self.records = json.loads((WORK/'selected.json').read_text())
        self.tiles = [np.load(r['cache'], mmap_mode='r') for r in self.records]
    def frame(self, t: float) -> np.ndarray:
        wall = np.zeros((GRID*180, GRID*320, 3), dtype=np.uint8)
        for index, data in enumerate(self.tiles):
            steps = max(1, len(data)-1)
            phase = (t*self.records[index]['playback_fps']+index*7)%(2*steps)
            position = min(phase, 2*steps-phase)
            lower, upper = min(int(position), len(data)-1), min(int(position)+1, len(data)-1)
            blend = position-int(position)
            tile = cv2.addWeighted(data[lower], 1-blend, data[upper], blend, 0)
            y, x = (index//GRID)*180, (index%GRID)*320
            wall[y+2:y+178, x+2:x+318] = tile[2:178, 2:318]
        # Show the entire 15-by-15 wall from the first frame. Only its source
        # clips move; the composition has no zoom, pan, or frozen overview.
        return cv2.resize(wall, SIZE, interpolation=cv2.INTER_AREA)


def encode_mosaic() -> None:
    mosaic = Mosaic()
    encoder = Encoder(OUT/'mosaic.mp4', crf=25)
    review = Image.new('RGB', (1920, 1140), '#101824')
    draw, font = ImageDraw.Draw(review), ImageFont.truetype(FONT, 24)
    review_times = [0, .75, 1.5, 2.75]
    for index in range(round(FPS*MOSAIC_DURATION)):
        frame = mosaic.frame(index/FPS)
        encoder.write(frame)
        if index/FPS in review_times:
            slot = review_times.index(index/FPS)
            x, y = slot%2*960, slot//2*570
            thumb = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            review.paste(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)), (x, y))
            draw.text((x+8, y+542), f'{index/FPS:.0f}s', font=font, fill='white')
        if index == round(FPS*(MOSAIC_DURATION-1)):
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(OUT/'mosaic-poster.webp', quality=94)
        if index%(FPS*2) == 0:
            print(f'MOSAIC {index/FPS:.0f}/{MOSAIC_DURATION:.0f}s', flush=True)
    encoder.close()
    review.save(WORK/'mosaic-review.jpg', quality=94)
    print('MOSAIC COMPLETE', flush=True)


@lru_cache(maxsize=12)
def scene_image(path: str) -> np.ndarray:
    return read(Path(path), *SIZE)


def timeline(scenes: list[dict], start: float = 0.) -> float:
    for record in scenes:
        default_duration = 4. if record['kind'] == 'mission_still' else 3.8 if record['collection'] == 'LuSNAR' else 2.65
        duration = float(record.get('duration', record.get('duration_seconds', default_duration)))
        record.update(start_seconds=round(start, 3), duration_seconds=duration,
                      startSeconds=round(start, 3), endSeconds=round(start+duration, 3))
        start += duration-CROSSFADE
    duration = round((start+CROSSFADE)*FPS)/FPS
    scenes[-1]['endSeconds'] = duration
    return duration


def wall_scene() -> dict:
    return {'id': 'video-wall', 'planet': 'wall', 'labelEn': 'Dataset', 'labelZh': 'Dataset',
            'kind': 'video_mosaic', 'start_seconds': 0., 'startSeconds': 0.,
            'endSeconds': MOSAIC_DURATION, 'duration_seconds': MOSAIC_DURATION}


def encode_journey() -> list[dict]:
    scenes = [record for record in json.loads((WORK/'hero-selected.json').read_text())
              if record['planet'] in {'moon', 'mars'}]
    duration = timeline(scenes, MOSAIC_DURATION)
    def scene_frame(index: int, t: float) -> np.ndarray:
        record, source = scenes[index], scenes[index]['frame_paths']
        if record['kind'] == 'mission_still':
            # Sparse real mission observation: zoom does not imply driving.
            frame = scene_image(source[0])
            scale = 1.01+.035*max(0, t)/record['duration_seconds']
            zoom = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            y, x = (zoom.shape[0]-SIZE[1])//2, (zoom.shape[1]-SIZE[0])//2
            return zoom[y:y+SIZE[1], x:x+SIZE[0]].copy()
        playback = min(15., max(2., (len(source)-1)/record['duration_seconds']))
        position = min(len(source)-1, max(0., t)*playback)
        # Preserve a sharp real observation within each segment. Blending
        # sparse source frames makes shelves, rails, and foliage appear doubled.
        return scene_image(source[min(round(position), len(source)-1)]).copy()
    encoder = Encoder(OUT/'journey.mp4', crf=27)
    # Open with the dataset wall. It shares the same media timeline as the
    # Moon/Mars scenes and needs no second player or source swap.
    mosaic = Mosaic()
    wall_review = Image.new('RGB', (1920, 1080), '#101824')
    for n in range(round(FPS*MOSAIC_DURATION)):
        frame = mosaic.frame(n/FPS)
        encoder.write(frame)
        if n == 0:
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(OUT/'journey-poster.webp', quality=93)
        if n/FPS in [0, .75, 1.5, 2.75]:
            slot = [0, .75, 1.5, 2.75].index(n/FPS)
            wall_review.paste(Image.fromarray(cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2RGB)), (slot%2*960, slot//2*540))
        if n == round(FPS*(MOSAIC_DURATION-1)):
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(OUT/'mosaic-poster.webp', quality=93)
        if n%(FPS*2) == 0:
            print(f'WALL {n/FPS:.0f}/{MOSAIC_DURATION:.0f}s', flush=True)
    review = Image.new('RGB', (4*480, math.ceil(len(scenes)/4)*294), '#0a111d')
    draw, font = ImageDraw.Draw(review), ImageFont.truetype(FONT, 18)
    starts = [record['startSeconds'] for record in scenes]
    captured = set()
    for n in range(round(FPS*(duration-MOSAIC_DURATION))):
        t = MOSAIC_DURATION+n/FPS
        index = min(len(scenes)-1, int(np.searchsorted(starts, t, side='right'))-1)
        local = t-starts[index]
        frame = scene_frame(index, local)
        if index > 0 and local < CROSSFADE:
            previous = scene_frame(index-1, t-starts[index-1])
            alpha = local/CROSSFADE
            alpha = alpha*alpha*(3-2*alpha)
            frame = cv2.addWeighted(previous, 1-alpha, frame, alpha, 0)
        # Responsive bilingual labels are rendered by the website, not burned
        # into the hero where object-fit could crop them on mobile screens.
        encoder.write(frame)
        if index not in captured and local >= scenes[index]['duration_seconds']/2:
            captured.add(index)
            x, y = index%4*480, index//4*294
            thumb = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            review.paste(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)), (x, y))
            draw.text((x+7, y+272), scenes[index]['labelEn'], font=font, fill='#f1f7fd')
        if n%(FPS*2) == 0:
            print(f'JOURNEY {t:.0f}/{duration:.2f}s | {scenes[index]["labelEn"]}', flush=True)
    encoder.close()
    wall_review.save(WORK/'mosaic-review.jpg', quality=93)
    scenes.insert(0, wall_scene())
    review.save(WORK/'journey-review.jpg', quality=93)
    print('JOURNEY COMPLETE', flush=True)
    return scenes


def public_record(record: dict) -> dict:
    allowed = {'id', 'source', 'collection', 'trajectory', 'first_index', 'frame_count',
               'trajectory_frame_count', 'first_frame', 'last_frame', 'frame_names', 'relative_path',
               'kind', 'labelEn', 'labelZh', 'grid_index', 'playback_fps',
               'start_seconds', 'duration_seconds', 'startSeconds', 'endSeconds', 'planet', 'split', 'editorial_note'}
    return {key: value for key, value in record.items() if key in allowed}


def write_manifest(scenes: list[dict], records: list[dict] | None = None) -> None:
    if records is None:
        records = json.loads((WORK/'selected.json').read_text())
    duration = scenes[-1]['endSeconds']
    media_hash = hashlib.sha256()
    for name in ['journey.mp4', 'journey-poster.webp', 'mosaic-poster.webp']:
        with (OUT/name).open('rb') as asset:
            for chunk in iter(lambda: asset.read(1024*1024), b''):
                media_hash.update(chunk)
    sources = [{'name': name, 'collection': 'NavAnywhere', 'clip_count': sum(r['source'] == name for r in records)}
               for name in dict.fromkeys(r['source'] for r in records)]
    chapters = []
    labels = {'moon': ('Moon', 'Moon'), 'mars': ('Mars', 'Mars'), 'wall': ('Dataset', 'Dataset')}
    for planet in ['wall', 'moon', 'mars']:
        members = [r for r in scenes if r['planet'] == planet]
        end = next((r['startSeconds'] for r in scenes if r['startSeconds'] > members[-1]['startSeconds']), duration)
        chapters.append({'planet': planet, 'labelEn': labels[planet][0], 'labelZh': labels[planet][1],
                         'startSeconds': members[0]['startSeconds'], 'endSeconds': end})
    wall = next(r for r in scenes if r['planet'] == 'wall')
    doc = {'version': 5, 'cacheVersion': f'fixed-wall-moon-mars-v5-{media_hash.hexdigest()[:12]}',
           'journey': {'src': 'assets/hero/journey.mp4', 'poster': 'assets/hero/journey-poster.webp',
                       'durationSeconds': duration, 'width': SIZE[0], 'height': SIZE[1], 'fps': FPS,
                       'perspective': 'first-person', 'order': ['Dataset video wall', 'Moon', 'Mars']},
           'mosaic': {'src': 'assets/hero/journey.mp4', 'poster': 'assets/hero/mosaic-poster.webp',
                      'startSeconds': wall['startSeconds'], 'endSeconds': wall['endSeconds'],
                      'durationSeconds': wall['duration_seconds'], 'width': SIZE[0], 'height': SIZE[1], 'fps': FPS,
                      'rows': GRID, 'columns': GRID, 'composition': 'fixed full-wall overview'},
           'counts': {'sources': len(sources), 'clips': len(records), 'tiles': GRID*GRID, 'journeyScenes': len(scenes)-1},
           'sources': sources,
           'chapters': chapters,
           'description': 'One continuous film opens with a fixed, fully visible 225-video dataset wall for three seconds, then travels through two Moon scenes and two Mars scenes.',
           'disclosure': 'Source-data showcase, not model predictions. Moon: LuSNAR simulation. Mars: SynMars-TW synthetic rover views from contiguous left-camera route frames. Playback speed is editorial; original camera timing and action ground truth are not implied.',
           'disclosureZh': 'Source-data showcase, not model predictions. Moon: LuSNAR simulation. Mars: SynMars-TW synthetic rover views from contiguous left-camera route frames. Playback speed is editorial; original camera timing and action ground truth are not implied.',
           'processing': ['Source frames are center-cropped to 16:9 and resized; presentation speed differs from capture timing.',
                          'Journey scenes remain first-person in Moon, Mars order, using nearest original source frames within each scene and brief cross-dissolves only between planetary scenes. Visual selection does not establish matched ground-truth actions.',
                          'The 15 by 15 wall contains 225 distinct trajectories from all 15 NavAnywhere sources. Short clips play forward then backward to soften loops.',
                          'The film opens with a three-second fixed overview of the complete moving dataset wall, followed directly by Moon and Mars. The wall has no zoom, pan, or burned-in labels. Bilingual journey names and chapter navigation are synchronized by the website.',
                          'All CASIA-Nav office trajectories are excluded. Four new outdoor trajectories preserve 225 distinct wall clips.',
                          'SynMars-TW is a synthetic Blender dataset, not footage captured on Mars. Each selected segment keeps one left-camera route and contiguous frame indices.'],
           'journeyScenes': [public_record(r) for r in scenes], 'clips': [public_record(r) for r in records]}
    MANIFEST.write_text(json.dumps(doc, indent=2, ensure_ascii=False)+'\n')


def render() -> None:
    global OUT
    destination = OUT
    OUT = WORK/'exports'
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        scenes = encode_journey()
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
            '-of', 'json', str(OUT/'journey.mp4')]))['streams'][0]
        expected_frames = round(scenes[-1]['endSeconds']*FPS)
        if (probe['width'], probe['height'], probe['r_frame_rate'], int(probe['nb_frames'])) != (*SIZE, f'{FPS}/1', expected_frames):
            raise ValueError(f'Unexpected rendered film format or frame count: {probe}')
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['journey.mp4', 'journey-poster.webp', 'mosaic-poster.webp']:
            temporary = destination/f'.{name}.pending'
            shutil.copy2(OUT/name, temporary)
            temporary.replace(destination/name)
    finally:
        OUT = destination
    write_manifest(scenes)
    print('RENDER COMPLETE', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['sample', 'prepare', 'render', 'mosaic', 'journey'])
    stage = parser.parse_args().stage
    print(f'START pid={os.getpid()} stage={stage} work={WORK}', flush=True)
    if stage == 'sample': sample()
    elif stage == 'prepare': prepare()
    elif stage == 'render': render()
    elif stage == 'mosaic': encode_mosaic()
    elif stage == 'journey': write_manifest(encode_journey())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit a selected gallery against original sources and saved inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from validate_go2_artifacts import probe, read, require, sha256, validate
from render_go2_gallery import validate_duration


def decode(path):
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
                    '-map', '0:v:0', '-an', '-threads', '2', '-f', 'null', '-'], check=True)


def public_metadata(value):
    """Expected public projection: private keys omitted, absolute paths basename-only."""
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items() if not key.startswith('_')}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    return Path(value).name if isinstance(value, str) and value.startswith('/') else value


def validate_clip_report(clip_report, spec, windows, selected, evaluation,
                         require_clip_mean=True):
    """Bind public plan data to windows and source result files, not another report."""
    label = spec['id']
    episode = Path(spec['episode_root'])
    require(clip_report['id'] == label and clip_report['episode'] == spec['episode_id'] and
            clip_report['start_s'] == spec['start_s'] and clip_report['end_s'] == spec['end_s'] and
            clip_report['goal_interval_s'] == windows['horizon'] / windows['fps'],
            f'{label}: report identity/timing')
    require(clip_report['evaluation_windows_sha256'] == sha256(evaluation / 'windows.json'),
            f'{label}: report windows checksum')
    require(clip_report['alignment_sha256'] == sha256(episode / 'alignment_manifest.json'),
            f'{label}: report alignment checksum')
    require(clip_report['alignment'] == public_metadata(read(episode / 'alignment_manifest.json').get('alignment', {})),
            f'{label}: report alignment differs')
    require(set(clip_report['model_protocols']) == {'opennwm', 'nwm'}, f'{label}: report model protocols')
    require([p['id'] for p in clip_report['plans']] == [row['id'] for row in selected],
            f'{label}: report plan coverage/order')
    if require_clip_mean or 'metric_aggregation' in clip_report:
        require(clip_report.get('metric_aggregation') == {
            'method': 'arithmetic_mean_of_complete_local_plans',
            'plan_count': len(selected), 'display': 'constant_for_entire_clip',
            'metrics': ['ate_m', 'rpe_m'], 'units': 'meters'},
            f'{label}: clip metric aggregation contract')
    results = {model: [] for model in ('opennwm', 'nwm')}
    for plan, row in zip(clip_report['plans'], selected):
        for key in ('start_time_s', 'end_time_s', 'reference_xy_m', 'reference_world_xy_m'):
            require(plan[key] == row[key], f'{label}/{row["id"]}: report {key} differs')
        require(set(plan['models']) == set(results), f'{label}/{row["id"]}: report models differ')
        for model in results:
            path = evaluation / 'outputs' / model / f'{row["id"]:06d}' / 'result.json'
            result = read(path)
            published = plan['models'][model]
            require(published['source_result_sha256'] == sha256(path),
                    f'{label}/{row["id"]}/{model}: report result checksum')
            for key in ('metrics', 'planned_xy_m', 'planned_yaw_rad'):
                require(published[key] == result[key], f'{label}/{row["id"]}/{model}: report {key} differs')
            results[model].append(result)
    require(set(clip_report['metrics']) == set(results), f'{label}: report summary models')
    for model, rows in results.items():
        require(clip_report['model_protocols'][model] ==
                public_metadata(read(evaluation / 'outputs' / model / 'protocol.json')),
                f'{label}/{model}: report protocol differs')
        require(set(clip_report['metrics'][model]) == {'ate_m', 'rpe_m'}, f'{label}: report summary metrics')
        for metric in ('ate_m', 'rpe_m'):
            np.testing.assert_allclose(clip_report['metrics'][model][metric],
                                       np.mean([r['metrics'][metric] for r in rows]), rtol=1e-10, atol=1e-12,
                                       err_msg=f'{label}/{model}: report summary {metric}')
    return results


def validate_gallery_coverage(plan, manifest, report):
    """Derive coverage from the frozen selection, without a fixed route count."""
    require(manifest['version'] in (2, 3) and manifest['status'] == 'ready', 'Gallery is not ready')
    if manifest['version'] >= 3:
        require(report.get('schema_version') == 3, 'Gallery report schema must match v3 manifest')
    require('full' not in manifest, 'Full-route comparison must not be in gallery')
    specs = {s['id']: s for s in plan['clips']}
    public = {c['id']: c for c in manifest['clips']}
    require(bool(specs), 'Gallery selection is empty')
    require(len(specs) == len(plan['clips']), 'Duplicate selected clips')
    require(len(public) == len(manifest['clips']) == len(specs), 'Clip count or duplicate mismatch')
    require(set(public) == set(specs), 'Manifest differs from selected clips')
    reports = {c['id']: c for c in report['clips']}
    require(len(reports) == len(report['clips']) and set(reports) == set(specs),
            'Report clip coverage or duplicate mismatch')
    episodes = set(s['episode_id'] for s in specs.values())
    require(set(report['episodes']) == episodes and len(report['episodes']) == len(episodes),
            'Report episode count/coverage differs from selection')
    for clip_id, spec in specs.items():
        require(public[clip_id]['episode'] == spec['episode_id'] == reports[clip_id]['episode'],
                f'{clip_id}: selected episode differs')
        if 'route' in spec:
            require(public[clip_id].get('route_id') == f"route_{spec['route']:02d}",
                    f'{clip_id}: route differs from selection')
    if 'routes' in plan:
        require(manifest.get('routes') == plan['routes'], 'Route metadata differs from selection')
    if 'display_clip_count' in manifest.get('selection', {}):
        require(manifest['selection']['display_clip_count'] == len(specs), 'Display clip count differs')
    return specs, public, reports


def probe_silent(path):
    info = probe(path)
    streams = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'json', str(path)]))
    require([stream['codec_type'] for stream in streams['streams']] == ['video'],
            f'Expected silent, single-video media: {path}')
    return info


def audit(root, website=None, decode_media=True, plan_path=None, gallery=None, publication=None):
    require(not publication or website is not None, 'A publication receipt requires a website path')
    plan = read(plan_path or root / 'gallery_plan.json')
    gallery = gallery or root / 'gallery'
    manifest = read(gallery / 'website_manifest.json')
    report = read(gallery / 'report.json')
    inventory = read(gallery / 'inventory.json')
    specs, public, reports = validate_gallery_coverage(plan, manifest, report)
    require('/file_system/' not in json.dumps([manifest, report]), 'Private path in public metadata')
    summary = {'episodes': {}, 'evaluations': {}, 'clips': {}}
    for spec in specs.values():
        episode = Path(spec['episode_root'])
        alignment = read(episode / 'alignment_manifest.json')
        if spec['episode_id'] not in summary['episodes']:
            archive = Path(alignment['source_archive'])
            require(sha256(archive) == alignment['source_archive_sha256'], 'Original archive checksum changed')
            aligned = [probe_silent(Path(alignment[k])) for k in ('ego_video', 'third_video')]
            require(abs(aligned[0]['duration_s'] - aligned[1]['duration_s']) < .001,
                    'Aligned camera durations differ')
            require(aligned[0]['nb_frames'] == aligned[1]['nb_frames'], 'Aligned camera frame counts differ')
            require(abs(aligned[0]['duration_s'] - alignment['aligned_duration_s']) < .08,
                    'Aligned duration differs from metadata')
            summary['episodes'][spec['episode_id']] = {
                'original_archive_sha256': alignment['source_archive_sha256'],
                'aligned_media': aligned,
                'alignment': alignment['alignment'],
            }
        evaluation = Path(spec['evaluation_root'])
        key = f"{spec['episode_id']}/{evaluation.name}"
        if key not in summary['evaluations']:
            summary['evaluations'][key] = validate(episode, evaluation_name=str(evaluation))
        windows = read(evaluation / 'windows.json')
        item = public[spec['id']]
        duration = spec['end_s'] - spec['start_s']
        validate_duration(spec)
        require(item['duration_s'] == duration, 'Clip duration metadata differs')
        selected = [r for r in windows['windows'] if r['id'] in item['evaluated_goal_ids']]
        selected.sort(key=lambda r: r['start_time_s'])
        require(len(selected) == len(item['evaluated_goal_ids']), 'Missing selected goals')
        require(selected[0]['start_time_s'] == spec['start_s'] and selected[-1]['end_time_s'] == spec['end_s'],
                'Clip boundaries do not match complete plans')
        require(all(a['end_time_s'] == b['start_time_s'] for a, b in zip(selected, selected[1:])),
                'Gap between plans')
        require(item['goal_interval_s'] == windows['horizon'] / windows['fps'], 'Planning interval mismatch')
        clip_report = read(gallery / spec['id'] / 'metrics.json')
        require(clip_report == reports[spec['id']],
                'Combined and individual reports differ')
        source_results = validate_clip_report(clip_report, spec, windows, selected, evaluation,
                                             require_clip_mean=manifest['version'] >= 3)
        if manifest['version'] >= 3 or 'metric_aggregation' in item:
            require(item.get('metric_aggregation') == clip_report.get('metric_aggregation'),
                    'Manifest metric aggregation differs from report')
        for model in ('opennwm', 'nwm'):
            results = source_results[model]
            for metric in ('ate_m', 'rpe_m'):
                np.testing.assert_allclose(item['metrics'][model][metric],
                                           np.mean([r['metrics'][metric] for r in results]), rtol=1e-10)
        exported = list(csv.DictReader((gallery / spec['id'] / 'trajectories.csv').open()))
        require(len(exported) == len(selected) * 3 * (windows['horizon'] + 1), 'Trajectory export incomplete')
        for row in selected:
            for model in ('gt', 'opennwm', 'nwm'):
                expected = row['reference_xy_m'] if model == 'gt' else read(
                    evaluation / 'outputs' / model / f"{row['id']:06d}" / 'result.json')['planned_xy_m']
                actual = [[float(v['forward_m']), float(v['left_m'])] for v in exported
                          if int(v['window_id']) == row['id'] and v['series'] == model]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        media = {}
        for filename in ('comparison.mp4', 'gt_ego.mp4', 'gt_third_person.mp4', 'opennwm.mp4', 'nwm.mp4'):
            path = gallery / spec['id'] / filename
            info = probe_silent(path)
            require(abs(info['duration_s'] - duration) < .08, f'Media duration mismatch: {path}')
            require(info['codec_name'] == 'h264' and info['pix_fmt'] == 'yuv420p', f'Incompatible media: {path}')
            if filename == 'comparison.mp4':
                require((info['width'], info['height']) == (1920, 1080), 'Comparison must be 1080p')
                rate = report['visualization']['fps']
                require(int(info['nb_frames']) == round(duration * rate), 'Comparison frame count mismatch')
            if decode_media:
                decode(path)
            media[filename] = info
        summary['clips'][spec['id']] = {'duration_s': duration, 'goal_interval_s': item['goal_interval_s'],
                                       'plans': len(selected), 'metrics': item['metrics'], 'media': media}
    for relative, entry in inventory.items():
        path = gallery / relative
        require(path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'],
                f'Gallery checksum mismatch: {relative}')
    if website:
        web_manifest = read(website / 'content/go2.json')
        expected_manifest = json.loads(json.dumps(manifest))
        publication_report = None
        if publication:
            publication_report = read(publication)
            require(publication_report['passed'], 'Web publication did not pass')
            require(publication_report.get('published'), 'Web publication has not been copied to the website')
            require(Path(publication_report['source']).resolve() == gallery.resolve(),
                    'Web publication uses a different source gallery')
            require(publication_report['source_manifest']['sha256'] == sha256(gallery / 'website_manifest.json'),
                    'Web publication used a different source manifest')
            require(any(Path(row['website']).resolve() == website.resolve() and row.get('passed')
                        for row in publication_report['websites']), 'Website was not validated by publication')
            for item in expected_manifest['clips']:
                prefix = f'assets/go2/{item["id"]}'
                item['downloads'] = {'metrics': f'{prefix}/metrics.json',
                                     'trajectories': f'{prefix}/trajectories.csv'}
        require(web_manifest == expected_manifest, 'Website manifest differs')
        require(read(website / 'assets/go2/report.json') == report, 'Website report differs')
        for item in web_manifest['clips']:
            for relative in (item['video'], item['poster'], *item['downloads'].values()):
                local = gallery / Path(relative).relative_to('assets/go2')
                published = website / relative
                if publication_report and relative == item['video']:
                    record = next(c for c in publication_report['clips'] if c['clip'] == item['id'])
                    require(record['passed'] and record['fully_decoded'], 'Web film was not fully validated')
                    require(record['master']['sha256'] == sha256(local), 'Web film master has changed')
                    require(record['output']['sha256'] == sha256(published), 'Web film has changed')
                    info = probe_silent(published)
                    require((info['width'], info['height']) == (1920, 1080), 'Web film is not 1080p')
                    require(abs(info['duration_s'] - item['duration_s']) < .08, 'Web film timing differs')
                    require(int(info['nb_frames']) == round(item['duration_s'] * report['visualization']['fps']),
                            'Web film frame count differs')
                    if decode_media:
                        decode(published)
                else:
                    require(sha256(published) == sha256(local), f'Website asset differs: {relative}')
        summary['website_assets_match'] = True
        summary['web_encoding_validated'] = bool(publication_report)
    summary['passed'] = True
    summary['media_fully_decoded'] = decode_media
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('/file_system/nas/algorithm/dujun.nie/opennwm_go2_showcase_20261027'))
    ap.add_argument('--website', type=Path)
    ap.add_argument('--plan', type=Path, help='Defaults to ROOT/gallery_plan.json')
    ap.add_argument('--gallery', type=Path, help='Defaults to ROOT/gallery')
    ap.add_argument('--publication', type=Path, help='Receipt for separately encoded website videos')
    ap.add_argument('--output', type=Path, help='Defaults to ROOT/validation.json')
    ap.add_argument('--skip-decode', action='store_true')
    args = ap.parse_args()
    if args.publication and not args.website:
        ap.error('--publication requires --website')
    result = audit(args.root, args.website, not args.skip_decode, args.plan, args.gallery, args.publication)
    output = args.output or args.root / 'validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'clips': len(result['clips']),
                      'episodes': len(result['episodes']), 'evaluations': len(result['evaluations'])}))


if __name__ == '__main__':
    main()

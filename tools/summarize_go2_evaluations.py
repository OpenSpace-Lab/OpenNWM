#!/usr/bin/env python3
"""Summarize saved Go2 results without rerunning or altering inference."""
import argparse
import json
from pathlib import Path
from statistics import mean

from evaluate_go2_recording import write_json


def summarize(results, expected_ids):
    found = {result['id'] for result in results}
    expected = set(expected_ids)
    return {
        'count': len(results),
        'expected': len(expected),
        'complete': found == expected,
        'missing_ids': sorted(expected - found),
        'unexpected_ids': sorted(found - expected),
        'means': {key: mean(result['metrics'][key] for result in results)
                  for key in results[0]['metrics']} if results else {},
        'total_inference_window_seconds': sum(result['elapsed_s'] for result in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evaluation-name', default='evaluation_1s')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    episodes, pooled = [], {'opennwm': [], 'nwm': []}
    expected_total = 0
    for episode_root in sorted((args.root / 'episodes').iterdir()):
        directory = episode_root / args.evaluation_name
        windows = json.loads((directory / 'windows.json').read_text())
        expected_ids = [row['id'] for row in windows['windows']]
        expected_total += len(expected_ids)
        item = {'episode_id': episode_root.name,
                'evaluation_root': str(directory),
                'horizon_seconds': windows['goal_interval_s'],
                'duration_seconds': windows['evaluated_duration_s'],
                'models': {}, 'clips': []}
        validation_path = directory / 'validation.json'
        if validation_path.exists():
            validation = json.loads(validation_path.read_text())
            item['independent_validation'] = {
                'report': str(validation_path),
                'passed': validation.get('passed', False),
                'windows_per_model': validation.get('expected_windows'),
            }
        for model in pooled:
            results = [json.loads(path.read_text()) for path in sorted(
                (directory / 'outputs' / model).glob('*/result.json'))]
            item['models'][model] = summarize(results, expected_ids)
            pooled[model].extend(results)
            for clip in windows['selected_clips']:
                matching = next((value for value in item['clips'] if value['id'] == clip['id']), None)
                if matching is None:
                    matching = {'id': clip['id'], 'label': clip['label'],
                                'start_s': clip['start_s'], 'duration_s': clip['duration_s'],
                                'models': {}}
                    item['clips'].append(matching)
                clip_ids = {row['id'] for row in windows['windows'] if clip['id'] in row['clip_ids']}
                matching['models'][model] = summarize(
                    [result for result in results if result['id'] in clip_ids], clip_ids)
        episodes.append(item)
    complete = all(value['complete'] for episode in episodes for value in episode['models'].values())
    if not complete and not args.allow_partial:
        raise SystemExit('Incomplete evaluation; use --allow-partial to write an explicitly partial summary')
    models = {}
    for model, results in pooled.items():
        # Window IDs repeat between episodes: aggregate the metrics, not the IDs.
        models[model] = dict(count=len(results), expected=expected_total,
            means={key: mean(result['metrics'][key] for result in results)
                   for key in results[0]['metrics']} if results else {},
            total_inference_window_seconds=sum(result['elapsed_s'] for result in results))
    result = dict(complete=complete, evaluation_name=args.evaluation_name,
        horizon_seconds=sorted({episode['horizon_seconds'] for episode in episodes}),
        duration_seconds=sum(episode['duration_seconds'] for episode in episodes),
        aggregation='Unweighted mean over independent local-plan windows; no per-clip selection by model output.',
        reference='Recorded Go2 local odometry; not independent motion capture.',
        scope='Offline image-goal CEM plans with recorded context resets; no robot execution.',
        models=models, episodes=episodes)
    result['independent_validation_complete'] = all(
        episode.get('independent_validation', {}).get('passed', False)
        for episode in episodes)
    quality_path = args.root / 'quality_selection.json'
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        result['quality_selection'] = {
            'record': str(quality_path),
            'status': quality['status'],
            'selected_horizon_seconds': quality.get('selected_horizon_s'),
            'conclusion': quality.get('final_conclusion', quality.get('provisional_reason')),
            'comparison_limitations': quality['comparison_limitations'],
        }
    output = args.root / ('evaluation_summary.json' if complete else 'evaluation_summary_partial.json')
    write_json(output, result)
    print(json.dumps(dict(output=str(output), complete=complete, models=models), indent=2))


if __name__ == '__main__':
    main()

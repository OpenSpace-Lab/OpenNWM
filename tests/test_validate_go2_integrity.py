"""CPU regression checks for CEM provenance and public-report integrity."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import validate_go2_artifacts as artifacts
import validate_go2_gallery as gallery


class CemIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(model='opennwm', updates=1, candidates=2, elites=1, repeats=1)
        self.deltas = np.zeros((4, 3), dtype=np.float32)
        second = np.array([[.2, 0, 0]] * 4, dtype=np.float32)
        second[-1, 2] = np.pi / 2
        self.audit = dict(gt_pose_used_for_optimization=False, iterations=[dict(
            candidate_parameters=[[0, 0, 0], [.2, 0, .5]],
            candidate_deltas_normalized=[self.deltas.tolist(), second.tolist()],
            candidate_losses_by_repeat=[[.25, .5]], mean_candidate_losses=[.25, .5],
            elite_candidate_ids=[0])], final_parameters=[0, 0, 0],
            final_planner_deltas=self.deltas.tolist(), model_conditioning=[[1.25, 0, 0]] * 4)
        self.raw = dict(planner_deltas=self.deltas.copy(),
                        body_actions_waypoint_units=np.array([[1.25, 0, 0]] * 4),
                        model_conditioning=np.array([[1.25, 0, 0]] * 4))

    def check(self):
        artifacts.validate_cem(self.audit, self.raw, self.protocol, 4, 'fixture')

    def test_valid_actions_for_both_model_adapters(self):
        self.check()
        self.protocol['model'] = 'nwm'
        self.audit['model_conditioning'] = self.deltas.tolist()
        self.raw['model_conditioning'] = self.deltas.copy()
        self.check()

    def test_independent_formula_checks_initial_and_terminal_turns(self):
        deltas, body = artifacts.proposal_actions([[-1, .5, .5]], 4)
        angle = np.arctan2(2, -2.5)
        np.testing.assert_allclose(deltas[0], [[-1, .5, angle], [-1, .5, 0],
                                             [-1, .5, 0], [-1, .5, np.pi / 2]], atol=2e-7)
        np.testing.assert_allclose(body[0, :, :2], [[-2.5, 2]] * 4)

    def test_changed_elite_mean_cannot_leave_actual_actions_unchanged(self):
        # The earlier validator accepted this internally consistent parameter
        # change even though the saved rollout still used the old action.
        self.audit['iterations'][0]['candidate_parameters'][0][0] = 1
        self.audit['iterations'][0]['candidate_deltas_normalized'][0] = [[1, 0, 0]] * 4
        self.audit['final_parameters'][0] = 1
        self.audit['final_planner_deltas'] = [[1, 0, 0]] * 4
        with self.assertRaisesRegex(AssertionError, 'rollout final actions'):
            self.check()

    def test_raw_planner_and_body_each_bound_to_final_proposal(self):
        original = copy.deepcopy(self.raw)
        for field in ('planner_deltas', 'body_actions_waypoint_units'):
            with self.subTest(field=field):
                self.raw = copy.deepcopy(original)
                self.raw[field][0, 0] += 1
                with self.assertRaises(AssertionError):
                    self.check()

    def test_candidate_conversion_and_repeat_loss_changes_rejected(self):
        original = copy.deepcopy(self.audit)
        for field in ('candidate_deltas_normalized', 'candidate_losses_by_repeat'):
            with self.subTest(field=field):
                self.audit = copy.deepcopy(original)
                values = self.audit['iterations'][0][field]
                if field == 'candidate_deltas_normalized':
                    values[0][0][2] += 1
                else:
                    values[0][0] += .1
                with self.assertRaises(AssertionError):
                    self.check()

    def test_repeated_bf16_mean_rounding_is_allowed_but_wrong_mean_is_not(self):
        self.protocol['repeats'] = 3
        self.audit['iterations'][0]['candidate_losses_by_repeat'] = [[.25, .5], [.25, .5], [.251953125, .5]]
        self.check()
        self.audit['iterations'][0]['candidate_losses_by_repeat'][2][0] = .5
        with self.assertRaisesRegex(AssertionError, 'repeated loss mean'):
            self.check()

    def test_negative_elite_index_is_rejected(self):
        self.audit['iterations'][0]['elite_candidate_ids'] = [-2]
        with self.assertRaisesRegex(AssertionError, 'elite indices'):
            self.check()


class ReportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evaluation = self.root / 'evaluation'
        self.evaluation.mkdir()
        self.spec = dict(id='clip', episode_id='episode', episode_root=str(self.root), start_s=2, end_s=4)
        self.selected = [dict(id=i, start_time_s=i, end_time_s=i + 1,
                              reference_xy_m=[[0, 0], [.1, 0]],
                              reference_world_xy_m=[[4, 5], [4.1, 5]]) for i in (2, 3)]
        self.windows = dict(horizon=4, fps=4, windows=self.selected)
        self.write(self.evaluation / 'windows.json', self.windows)
        self.write(self.root / 'alignment_manifest.json', dict(alignment={'offset_s': -.1}))
        self.report = dict(id='clip', episode='episode', start_s=2, end_s=4, goal_interval_s=1,
            alignment={'offset_s': -.1}, alignment_sha256=artifacts.sha256(self.root / 'alignment_manifest.json'),
            evaluation_windows_sha256=artifacts.sha256(self.evaluation / 'windows.json'),
            metrics={}, model_protocols={}, plans=[],
            metric_aggregation={'method': 'arithmetic_mean_of_complete_local_plans',
                'plan_count': 2, 'display': 'constant_for_entire_clip',
                'metrics': ['ate_m', 'rpe_m'], 'units': 'meters'})
        for model in ('opennwm', 'nwm'):
            protocol = dict(model=model, checkpoint={'path': '/private/checkpoint.pth', 'sha256': 'checksum'})
            self.write(self.evaluation / 'outputs' / model / 'protocol.json', protocol)
            self.report['model_protocols'][model] = dict(model=model, checkpoint={'path': 'checkpoint.pth', 'sha256': 'checksum'})
            self.report['metrics'][model] = dict(ate_m=.1, rpe_m=.02)
        for row in self.selected:
            plan = {**copy.deepcopy(row), 'models': {}}
            for model in ('opennwm', 'nwm'):
                result = dict(metrics=dict(ate_m=.1, rpe_m=.02, endpoint_m=.2),
                              planned_xy_m=[[0, 0], [.2, 0]], planned_yaw_rad=[0, .1])
                path = self.evaluation / 'outputs' / model / f'{row["id"]:06d}' / 'result.json'
                self.write(path, result)
                plan['models'][model] = {**result, 'source_result_sha256': artifacts.sha256(path)}
            self.report['plans'].append(plan)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def check(self):
        return gallery.validate_clip_report(self.report, self.spec, self.windows, self.selected, self.evaluation)

    def test_unchanged_report_matches_independent_source_files(self):
        self.assertEqual(len(self.check()['opennwm']), 2)

    def test_stale_public_metrics_trajectory_yaw_and_checksum_rejected(self):
        original = copy.deepcopy(self.report)
        mutations = {
            'metrics': {'ate_m': 9, 'rpe_m': .02, 'endpoint_m': .2},
            'planned_xy_m': [[0, 0], [9, 0]],
            'planned_yaw_rad': [0, 9],
            'source_result_sha256': 'stale',
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                self.report = copy.deepcopy(original)
                self.report['plans'][0]['models']['opennwm'][key] = value
                # Keeping combined and individual copies identical is no longer
                # sufficient: every copied value is compared to source results.
                with self.assertRaises(AssertionError):
                    self.check()

    def test_report_plan_coverage_and_reference_are_bound_to_windows(self):
        original = copy.deepcopy(self.report)
        for change in ('missing', 'duplicate', 'time', 'reference'):
            with self.subTest(change=change):
                self.report = copy.deepcopy(original)
                if change == 'missing':
                    self.report['plans'].pop()
                elif change == 'duplicate':
                    self.report['plans'][1] = copy.deepcopy(self.report['plans'][0])
                elif change == 'time':
                    self.report['plans'][0]['start_time_s'] = 99
                else:
                    self.report['plans'][0]['reference_xy_m'][1][0] = 99
                with self.assertRaises(AssertionError):
                    self.check()

    def test_summary_protocol_and_provenance_cannot_be_stale(self):
        original = copy.deepcopy(self.report)
        for change in ('metrics', 'model_protocols', 'evaluation_windows_sha256', 'alignment_sha256'):
            with self.subTest(change=change):
                self.report = copy.deepcopy(original)
                if change == 'metrics':
                    self.report[change]['nwm']['ate_m'] = 9
                elif change == 'model_protocols':
                    self.report[change]['nwm']['checkpoint']['sha256'] = 'stale'
                else:
                    self.report[change] = 'stale'
                with self.assertRaises(AssertionError):
                    self.check()

    def test_missing_or_per_plan_metric_contract_is_rejected(self):
        original = copy.deepcopy(self.report)
        for field, value in [('method', 'weighted_mean'), ('plan_count', 1),
                             ('display', 'current_local_plan'), ('units', 'waypoint_units'),
                             ('metrics', ['ate_m'])]:
            with self.subTest(field=field):
                self.report = copy.deepcopy(original)
                self.report['metric_aggregation'][field] = value
                with self.assertRaisesRegex(AssertionError, 'aggregation contract'):
                    self.check()
        self.report = copy.deepcopy(original)
        self.report.pop('metric_aggregation')
        with self.assertRaisesRegex(AssertionError, 'aggregation contract'):
            self.check()
        # Historical v2 reports remain auditable against their source outputs.
        gallery.validate_clip_report(self.report, self.spec, self.windows, self.selected,
                                     self.evaluation, require_clip_mean=False)


class GalleryCoverageTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'clips': [dict(id=f'clip_{i}', episode_id=f'episode_{i}', route=i)
                               for i in range(1, 12)],
                     'routes': [dict(id=f'route_{i:02d}', episode=f'episode_{i}') for i in range(1, 12)]}
        self.manifest = dict(version=3, status='ready', routes=copy.deepcopy(self.plan['routes']),
            selection={'display_clip_count': 11}, clips=[
                dict(id=s['id'], episode=s['episode_id'], route_id=f"route_{s['route']:02d}")
                for s in self.plan['clips']])
        self.report = dict(schema_version=3, episodes=[s['episode_id'] for s in self.plan['clips']],
                           clips=copy.deepcopy(self.manifest['clips']))

    def check(self):
        return gallery.validate_gallery_coverage(self.plan, self.manifest, self.report)

    def test_eleven_routes_are_checked_against_plan_not_fixed_count(self):
        self.assertEqual(len(self.check()[0]), 11)
        self.manifest['version'] = 2
        self.report['schema_version'] = 2
        self.check()

    def test_missing_duplicate_or_wrong_episode_is_rejected(self):
        original = copy.deepcopy(self.report)
        for episodes in [original['episodes'][:-1], original['episodes'] + ['episode_1'],
                         original['episodes'][:-1] + ['unexpected_episode']]:
            with self.subTest(episodes=episodes):
                self.report = {**original, 'episodes': episodes}
                with self.assertRaisesRegex(AssertionError, 'episode count/coverage'):
                    self.check()

    def test_changed_route_count_and_report_schema_are_rejected(self):
        original = copy.deepcopy(self.manifest)
        for mutation in [dict(routes=[]), dict(selection={'display_clip_count': 8}),
                         dict(clips=original['clips'][:-1]), dict(version=4)]:
            with self.subTest(mutation=mutation):
                self.manifest = {**original, **mutation}
                with self.assertRaises(AssertionError):
                    self.check()
        self.manifest = original
        self.report['schema_version'] = 2
        with self.assertRaisesRegex(AssertionError, 'report schema'):
            self.check()


if __name__ == '__main__':
    unittest.main(verbosity=2)

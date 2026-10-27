"""Deterministic candidate constraints based only on completed measured actions."""
import math


class TurnReversalGuard:
    """Reject opposite in-place turns until 20 cm of measured translation.

    Stationary choices, cancelled proposals, goal changes and replans do not
    call observe and therefore cannot erase the last executed turn. Small yaw
    changes below 3 degrees are treated as measurement/straight-line drift.
    """
    release_distance_m = .20
    significant_turn_rad = math.pi / 60

    def __init__(self):
        self.last_turn = None
        self.translation_progress_m = 0.

    def snapshot(self):
        return {'last_turn': dict(self.last_turn) if self.last_turn else None,
                'translation_progress_m': self.translation_progress_m,
                'release_distance_m': self.release_distance_m,
                'significant_turn_rad': self.significant_turn_rad}

    def eligible(self, proposals):
        blocked = {}
        if self.last_turn:
            for name, proposal in proposals.items():
                yaw = proposal['nominal']['yaw_rad']
                if proposal['nominal']['distance_m'] == 0 and yaw * self.last_turn['direction'] < 0:
                    blocked[name] = {'rule': 'opposite_in_place_turn_without_translation',
                                     'nominal_yaw_rad': yaw,
                                     'previous_executed_turn': dict(self.last_turn)}
        return ({k: v for k, v in proposals.items() if k not in blocked},
                {'state': self.snapshot(), 'blocked': blocked,
                 'eligible': [k for k in proposals if k not in blocked]})

    def observe(self, proposal, execution, step):
        if execution.get('status') not in ('succeeded', 'stopped_at_reserve'):
            return None
        if any(execution.get(k) for k in ('error', 'budget_error', 'feedback_error')):
            return None
        if proposal['candidate'] == 'stationary':
            return None
        measured = execution.get('measured_m_rad')
        if (not isinstance(measured, (list, tuple)) or len(measured) != 3
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in measured)):
            return None
        dx, dy, yaw = measured
        if proposal['nominal']['distance_m'] == 0 and abs(yaw) >= self.significant_turn_rad:
            self.last_turn = {'candidate': proposal['candidate'], 'step': step,
                              'direction': 1 if yaw > 0 else -1, 'measured_yaw_rad': yaw}
            self.translation_progress_m = 0.
            return {'event': 'turn_recorded', 'state': self.snapshot()}
        # Arcs are not inverse rotations: allow them in both directions and
        # count their actual forward translation toward releasing the guard.
        if (self.last_turn and proposal['nominal']['distance_m'] > 0 and dx > 0):
            self.translation_progress_m += math.hypot(dx, dy)
            if self.translation_progress_m + 1e-9 >= self.release_distance_m:
                previous = self.snapshot()
                self.last_turn = None
                self.translation_progress_m = 0.
                return {'event': 'released_by_translation', 'previous': previous,
                        'state': self.snapshot()}
            return {'event': 'translation_progress', 'state': self.snapshot()}
        return None

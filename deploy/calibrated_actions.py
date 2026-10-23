"""Fixed 15-action contract and offline calibration; no robot/GPU imports."""
import copy
import hashlib
import json
import math

SPACING = .25
MODEL_HORIZON_SECONDS = .5
MODEL_FRAME_OFFSET = MODEL_HORIZON_SECONDS * 4
NORMALIZED_K = MODEL_FRAME_OFFSET / 128
SCHEMA = 'opennwm.fixed15.v1'
HARD_DISTANCE = .7
HARD_TURN = math.pi / 3
SCHEDULE_KEYS = ('vx_m_s', 'vy_m_s', 'yaw_rad_s', 'duration_s')


def candidate_specs():
    result = [{'name': 'stationary', 'distance_m': 0., 'yaw_rad': 0.}]
    for distance in (.25, .5):
        tag = str(int(distance * 100))
        result.append(dict(name='forward_' + tag, distance_m=distance, yaw_rad=0.))
        for angle, label in ((math.pi/12, 'pi12'), (math.pi/4, 'pi4')):
            for side, sign in (('left', 1), ('right', -1)):
                result.append(dict(name='%s_arc_%s_%s' % (side, tag, label),
                                   distance_m=distance, yaw_rad=sign*angle))
    for angle, label in ((math.pi/12, 'pi12'), (math.pi/4, 'pi4')):
        for side, sign in (('left', 1), ('right', -1)):
            result.append(dict(name='rotate_%s_%s' % (side, label), distance_m=0., yaw_rad=sign*angle))
    return result


def nominal_proposals():
    from jetson_navigate import twist_endpoint, timed_schedule
    result = {}
    for spec in candidate_specs():
        endpoint = twist_endpoint([spec['distance_m'], 0., spec['yaw_rad']])
        schedule = timed_schedule(endpoint)
        if spec['name'] == 'stationary':
            schedule['duration_s'] = .5  # Observation-only hold, never sent to Move.
        result[spec['name']] = dict(candidate=spec['name'], nominal=spec,
            metric_target=endpoint, raw_candidate=[endpoint[0]/SPACING, endpoint[1]/SPACING, endpoint[2]],
            schedule=schedule, hard_distance_m=HARD_DISTANCE, hard_turn_rad=HARD_TURN,
            mapped_path_length_m=spec['distance_m'])
    return result


def contract_hash(proposals=None):
    return hashlib.sha256(json.dumps(proposals if proposals is not None else nominal_proposals(), sort_keys=True).encode()).hexdigest()


def finite_vector(values, count):
    if (not isinstance(values, list) or len(values) != count
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
        raise ValueError('Expected %d finite numbers' % count)
    return values


def summarize_samples(name, samples, repeats):
    """All trials must be valid; rejected trials are never silently discarded."""
    if len(samples) < repeats:
        raise ValueError('%s: need %d completed trials' % (name, repeats))
    poses, horizons = [], []
    for sample in samples:
        if sample.get('status') not in ('succeeded', 'stopped_at_reserve'):
            raise ValueError(name + ': failed execution')
        if any(sample.get(k) for k in ('error', 'budget_error', 'feedback_error', 'avoidance_intervened')):
            raise ValueError(name + ': execution/feedback error')
        pose = finite_vector(sample.get('measured_m_rad'), 3)
        path = sample.get('measured_path_m')
        turn = sample.get('measured_accumulated_turn_rad')
        horizon = sample.get('observation_horizon_s')
        finite_vector([path, turn, horizon], 3)
        if not (0 <= path < HARD_DISTANCE and 0 <= turn < HARD_TURN and 0 < horizon <= 32):
            raise ValueError(name + ': motion bounds or observation duration invalid')
        if math.hypot(*pose[:2]) >= HARD_DISTANCE or abs(pose[2]) >= HARD_TURN:
            raise ValueError(name + ': endpoint exceeds hard bounds')
        if name == 'stationary':
            if math.hypot(*pose[:2]) > .01 or abs(pose[2]) > math.pi/90:
                raise ValueError('Stationary trial drifted; retry with robot stopped')
        poses.append(pose)
        horizons.append(horizon)
    mean = [sum(p[i] for p in poses)/len(poses) for i in range(3)]
    xy_spread = max(math.hypot(p[0]-mean[0], p[1]-mean[1]) for p in poses)
    yaw_spread = max(abs(p[2]-mean[2]) for p in poses)
    if xy_spread > .04 or yaw_spread > math.pi/60:
        raise ValueError('%s: not repeatable; max XY deviation %.4f m (limit 0.04 m), max yaw deviation %.5f*pi / %.3f deg (limit pi/60 / 3 deg)'
                         % (name, xy_spread, yaw_spread/math.pi, math.degrees(yaw_spread)))
    spec = next(s for s in candidate_specs() if s['name'] == name)
    if spec['distance_m'] and (math.hypot(*mean[:2]) < .2 or mean[0] <= 0):
        raise ValueError(name + ': insufficient/incorrect translation')
    if spec['yaw_rad'] and mean[2]*spec['yaw_rad'] <= 0:
        raise ValueError(name + ': incorrect turn direction')
    if spec['yaw_rad'] and abs(mean[2]) < math.pi/12:
        raise ValueError('%s: mean turn %.5f*pi / %.3f deg below pi/12 / 15 deg'
                         % (name, abs(mean[2])/math.pi, math.degrees(abs(mean[2]))))
    if spec['distance_m']:
        from jetson_navigate import body_twist
        if abs(math.hypot(*body_twist(mean)[:2])-spec['distance_m']) > .05:
            raise ValueError(name + ': distance differs from nominal by more than 0.05 m')
    if spec['yaw_rad'] and abs(mean[2]-spec['yaw_rad']) > math.pi/36 + 1e-9:
        raise ValueError(name + ': turn differs from nominal by more than pi/36')
    if not spec['distance_m'] and math.hypot(*mean[:2]) > .04:
        raise ValueError(name + ': excessive drift for rotation')
    if spec['distance_m'] and not spec['yaw_rad'] and abs(mean[1]) > .04:
        raise ValueError(name + ': excessive lateral drift for straight motion')
    if not spec['yaw_rad'] and abs(mean[2]) > math.pi/60:
        raise ValueError(name + ': excessive unintended turn')
    if name == 'stationary':
        mean = [0., 0., 0.]  # No velocity command; drift above tolerance is rejected.
    return dict(mean_m_rad=mean, model_action=[mean[0]/SPACING, mean[1]/SPACING, mean[2]],
                observation_horizon_s=sum(horizons)/len(horizons),
                max_xy_deviation_m=xy_spread, max_yaw_deviation_rad=yaw_spread, sample_count=len(samples))


def build_calibration(samples, repeats=3, proposals=None):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
        raise ValueError('At least three repetitions required')
    entries, problems = {}, []
    for name, proposal in (proposals or nominal_proposals()).items():
        trials = samples.get(name, [])
        entry = dict(proposal=proposal, samples=trials)
        try:
            entry['summary'] = summarize_samples(name, trials, repeats)
        except ValueError as exc:
            problems.append(str(exc))
        entries[name] = entry
    return dict(schema=SCHEMA, waypoint_spacing_m=SPACING, contract_sha256=contract_hash(),
                repeats=repeats, ready=not problems, problems=problems, actions=entries,
                timing_policy='fixed_model_horizon_0.5s',
                observation_timing_policy='measured_before_to_after_image_interval_for_records_only',
                execution_policy='fixed_timed_commands_with_existing_motion_guards_no_online_learning')


def load_calibration(path, require_ready=True, redo_candidates=()):
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA or data.get('waypoint_spacing_m') != SPACING:
        raise ValueError('Wrong calibration schema/waypoint spacing')
    legacy = nominal_proposals()
    for proposal in legacy.values():
        proposal.update(hard_distance_m=.5, hard_turn_rad=math.pi/4)
    migrate_limits = not require_ready and data.get('contract_sha256') == contract_hash(legacy)
    if (data.get('contract_sha256') != contract_hash() and not migrate_limits) or (require_ready and data.get('ready') is not True):
        raise ValueError('Calibration incomplete or candidate/control contract changed')
    expected = legacy if migrate_limits else nominal_proposals()
    if set(data.get('actions', {})) != set(expected):
        raise ValueError('Calibration must contain exactly the fixed 15 candidates')
    if redo_candidates and require_ready:
        raise ValueError('Candidate recollection is only allowed when resuming calibration')
    reset = {}
    for name in dict.fromkeys(redo_candidates):
        if name not in expected:
            raise ValueError('Unknown candidate to recollect: ' + name)
        reset[name] = len(data['actions'][name]['samples'])
        data['actions'][name]['samples'] = []
    # Recompute, rather than trusting editable ready flags or cached averages.
    proposals = {k: v['proposal'] for k, v in data['actions'].items()}
    for name, proposal in proposals.items():
        baseline = expected[name].copy()
        schedule = proposal.get('schedule', {})
        validate_schedule(schedule, stationary=name == 'stationary')
        baseline['schedule'] = schedule
        if baseline != proposal:
            raise ValueError('Candidate definition changed: ' + name)
    dropped = []
    if migrate_limits:
        for name, proposal in proposals.items():
            trials = data['actions'][name]['samples']
            # Old reserve-limited executions would stop later under the new limits.
            # Preserve only whole repeat sets completed by their timed commands.
            if any(sample.get('status') != 'succeeded' or
                   (name != 'stationary' and sample.get('controller', {}).get('completion_basis') != 'elapsed_command_time')
                   for sample in trials):
                if any(sample.get('status') == 'failed' for sample in trials):
                    raise ValueError('Failed sample cannot be migrated: ' + name)
                dropped.append(name)
                data['actions'][name]['samples'] = []
            proposal.update(hard_distance_m=HARD_DISTANCE, hard_turn_rad=HARD_TURN)
    checked = build_calibration({k: v['samples'] for k, v in data['actions'].items()}, data.get('repeats'), proposals)
    if require_ready and not checked['ready']:
        raise ValueError('; '.join(checked['problems']))
    for name, entry in data['actions'].items():
        if entry['samples']:
            summarize_samples(name, entry['samples'], len(entry['samples']))
        for sample in entry['samples']:
            if sample.get('proposal', {}).get('schedule') != proposals[name]['schedule']:
                raise ValueError('Trial was collected with a different schedule: ' + name)
    if reset:
        checked['recollection'] = {'discarded_from_new_run_only': reset, 'source_file_unchanged': True}
    if migrate_limits:
        checked['limit_migration'] = {'from_m_rad': [.5, math.pi/4],
            'to_m_rad': [HARD_DISTANCE, HARD_TURN], 'recollect_candidates': dropped,
            'source_contract_sha256': data['contract_sha256']}
    return checked


def calibrated_proposals(calibration):
    result = {}
    for name in nominal_proposals():
        entry = calibration['actions'][name]
        proposal = copy.deepcopy(entry['proposal'])
        summary = entry['summary']
        proposal.update(metric_target=summary['mean_m_rad'][:],
            effective_candidate=summary['model_action'][:], effective_path=[summary['model_action'][:]],
            observation_horizon_s=summary['observation_horizon_s'],
            model_horizon_seconds=MODEL_HORIZON_SECONDS, normalized_k=NORMALIZED_K,
            prediction_condition='offline_measured_mean_endpoint', calibration_summary=summary.copy())
        result[name] = proposal
    return result


def validate_schedule(schedule, stationary=False):
    if set(schedule) != set(SCHEDULE_KEYS):
        raise ValueError('Invalid schedule keys')
    vx, vy, yaw, duration = finite_vector([schedule[k] for k in SCHEDULE_KEYS], 4)
    if not (0 < duration <= 15) or math.hypot(vx, vy) > .120001 or abs(yaw) > .350001:
        raise ValueError('Schedule exceeds velocity/time limits')
    if stationary and (vx != 0 or vy != 0 or yaw != 0):
        raise ValueError('Stationary must never command motion')

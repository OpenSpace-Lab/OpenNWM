import copy
import math
from calibrated_actions import nominal_proposals, build_calibration
from jetson_navigate import twist_endpoint


def calibration_fixture():
    proposals = nominal_proposals()
    samples = {}
    for name, proposal in proposals.items():
        spec = proposal['nominal']
        distance = min(spec['distance_m'], .48)
        yaw = spec['yaw_rad']
        if yaw:
            yaw = math.copysign(math.radians(18 if abs(yaw) < .5 else 40), yaw)
        measured = twist_endpoint([distance, 0., yaw])
        sample = dict(status='succeeded', proposal=copy.deepcopy(proposal), measured_m_rad=measured,
                      measured_path_m=distance, measured_accumulated_turn_rad=abs(yaw),
                      observation_horizon_s=max(.5, proposal['schedule']['duration_s']+1.))
        samples[name] = [copy.deepcopy(sample) for _ in range(3)]
    return build_calibration(samples)

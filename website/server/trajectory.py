"""Validated top-down drawing to the model's current-body SE(2) controls."""
import math

FRAMES = 16
FPS = 4
PATH_UNITS = 12.0  # One canvas unit = 12 dataset waypoint-spacing units.

PRESETS = {
    "forward": [[0, 0], [0, 1]],
    "left": [[0, 0], [0, .2], [-.03, .4], [-.12, .6], [-.28, .8], [-.5, 1]],
    "right": [[0, 0], [0, .2], [.03, .4], [.12, .6], [.28, .8], [.5, 1]],
}


def validate_points(points):
    if not isinstance(points, list) or not 2 <= len(points) <= 512:
        raise ValueError("Draw a path with 2–512 points.")
    clean = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each path point must contain x and y.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
            raise ValueError("Path coordinates must be finite numbers.")
        x, y = map(float, point)
        if not (-1 <= x <= 1 and -.25 <= y <= 1.5):
            raise ValueError("Path is outside the drawing area.")
        if not clean or math.dist(clean[-1], (x, y)) > 1e-7:
            clean.append([x, y])
    if not clean or math.dist(clean[0], (0, 0)) > .025:
        raise ValueError("Start your path at the robot marker.")
    clean[0] = [0., 0.]
    length = sum(math.dist(a, b) for a, b in zip(clean, clean[1:]))
    if not .05 <= length <= 3:
        raise ValueError("Draw a path between 0.05 and 3 canvas units long.")
    return clean


def path_to_actions(points):
    """Arc-length resampling; canvas right/forward -> model forward/left.

    Translation is in dataset waypoint-spacing units, not meters. The real
    adapter owns min/max normalization. Heading follows the path tangent;
    initial heading is forward. No vertical component is inferred.
    """
    points = validate_points(points)
    lengths = [0.]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    poses = [(0., 0., 0.)]
    segment = 0
    for index in range(1, FRAMES + 1):
        distance = lengths[-1] * index / FRAMES
        while segment < len(points) - 2 and lengths[segment + 1] < distance:
            segment += 1
        a, b = points[segment:segment + 2]
        fraction = (distance - lengths[segment]) / (lengths[segment + 1] - lengths[segment])
        right = a[0] + fraction * (b[0] - a[0])
        forward = a[1] + fraction * (b[1] - a[1])
        yaw = math.atan2(-(b[0] - a[0]), b[1] - a[1])
        poses.append((forward * PATH_UNITS, -right * PATH_UNITS, yaw))
    actions = []
    for start, end in zip(poses, poses[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        c, s = math.cos(start[2]), math.sin(start[2])
        angle = end[2] - start[2]
        actions.append([c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(angle), math.cos(angle))])
    return actions


def actions_to_endpoints(actions):
    """Recover origin-frame endpoints for fixed-context direct prediction."""
    x = y = yaw = 0.
    endpoints = []
    for dx, dy, angle in actions:
        x += math.cos(yaw) * dx - math.sin(yaw) * dy
        y += math.sin(yaw) * dx + math.cos(yaw) * dy
        yaw += angle
        endpoints.append([x, y, math.atan2(math.sin(yaw), math.cos(yaw))])
    return endpoints

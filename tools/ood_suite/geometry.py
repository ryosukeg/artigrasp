"""Small wxyz SO(3) helpers, shared by target generation and goal execution."""

import math


def unit_quaternion(values):
    values = [float(value) for value in values]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected four finite quaternion components (wxyz)")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        raise ValueError("Quaternion has zero norm")
    return [value / norm for value in values]


def quaternion_product(left, right):
    w, x, y, z = unit_quaternion(left)
    a, b, c, d = unit_quaternion(right)
    return unit_quaternion([w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                            w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a])


def axis_angle_quaternion(axis, angle):
    axis = [float(value) for value in axis]
    if len(axis) != 3 or not all(math.isfinite(value) for value in axis):
        raise ValueError("Expected a finite 3D rotation axis")
    norm = math.sqrt(sum(value * value for value in axis))
    if norm < 1e-12 or not math.isfinite(angle):
        raise ValueError("Rotation axis must be nonzero and angle finite")
    sine = math.sin(angle / 2.0)
    return [math.cos(angle / 2.0)] + [sine * value / norm for value in axis]


def rotate_quaternion(reference, rotation):
    """World rotations multiply on the left; object-frame rotations on the right."""
    delta = rotation["quaternion_wxyz"]
    frame = rotation["frame"]
    if frame == "world":
        return quaternion_product(delta, reference)
    if frame == "object":
        return quaternion_product(reference, delta)
    raise ValueError("Rotation frame must be 'world' or 'object'")


def orientation_goal_pair(control, rotation=None):
    """Return (placement, lift); the unaffected stage keeps its paired control."""
    control = unit_quaternion(control)
    if rotation is None:
        return control[:], control[:]
    stage = rotation["stage"]
    if stage not in ("placement", "lift", "both"):
        raise ValueError("Rotation stage must be placement, lift, or both")
    changed = rotate_quaternion(control, rotation)
    return (changed if stage in ("placement", "both") else control[:],
            changed if stage in ("lift", "both") else control[:])

"""Two deliberately separate evaluations of the ArtiGrasp composed task.

The paper uses the final physical object pose.  ``public_*`` reproduces
compose_eval.py/ketchup_eval.py at repository revision 92fb307, including its
one-step-old observations, independent success latches, quaternion-index bug,
and clipped error summaries.  Do not substitute public success for paper Suc.T.

This module uses only the standard library so it can be tested without RaiSim.
"""

import math


PAPER_POSITION_THRESHOLD_M = 0.05
PAPER_ORIENTATION_THRESHOLD_RAD = 0.2
PAPER_ARTICULATION_THRESHOLD_RAD = 0.5
GLOBAL_STATE_SIZE = 138
OBJECT_POSITION = slice(122, 125)
OBJECT_QUATERNION = slice(125, 129)
OBJECT_ANGLE = 129
TARGET_POSITION = slice(130, 133)
TARGET_QUATERNION = slice(133, 137)
TARGET_ANGLE = 137


def _unit_quaternion(values):
    q = tuple(float(x) for x in values)
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError("Quaternion must contain four finite wxyz values")
    norm = math.sqrt(sum(x * x for x in q))
    if norm < 1e-12:
        raise ValueError("Quaternion has zero norm")
    return tuple(x / norm for x in q)


def quaternion_geodesic_rad(actual, target):
    """SO(3) geodesic, treating q and -q as the same orientation."""
    a, b = _unit_quaternion(actual), _unit_quaternion(target)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2.0 * math.acos(min(1.0, max(0.0, dot)))


def public_euler_y_rad(quaternion):
    """Exact middle Euler component from helper.rotations.quat2euler."""
    w, x, y, z = _unit_quaternion(quaternion)
    m02 = 2.0 * (x * z + w * y)
    m12 = 2.0 * (y * z - w * x)
    m22 = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(m02, math.sqrt(m22 * m22 + m12 * m12))


def wilson_interval(successes, trials, z=1.959963984540054):
    """Wilson 95% interval (proportions); missing results return None."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Require 0 <= successes <= trials")
    if not trials:
        return None
    p = successes / float(trials)
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _invalid(reason):
    return {
        "valid": False,
        "invalid_reason": str(reason),
        "paper_success": False,
        "public_success": False,
        "paper_position_error_m": None,
        "paper_orientation_error_rad": None,
        "paper_articulation_error_rad": None,
        "public_position_error_m": None,
        "public_orientation_error_rad": None,
        "public_articulation_error_rad": None,
        "public_position_error_clipped_m": None,
        "public_orientation_error_clipped_rad": None,
        "public_articulation_error_clipped_rad": None,
    }


def evaluate_episode(states, spec, timing):
    """Evaluate one rollout and return JSON-serializable flat metric fields.

    ``states[t]`` is the physical global state after t completed env.step calls,
    with states[0] taken just after reset.  All 138 C++ fields are required.
    ``timing`` contains grasp_steps, transit_steps, arti_steps, and optionally
    protocol (standard or ketchup); its total must match len(states)-1.
    ``spec`` contains object and optionally explicit target_position_m,
    target_quaternion_wxyz, target_articulation_rad, initial_articulation_rad,
    target_euler_y_rad.  The explicit target values describe the final task,
    including for an unreachable requested target.  No angle wrapping/clamping
    is applied to articulation errors: this joint is limited, not continuous.

    Invalid/nonfinite/incomplete rollouts count as failures.  None-valued errors
    let the report show missing measurements instead of fabricated numeric errors.
    """
    try:
        g = int(timing["grasp_steps"])
        t = int(timing["transit_steps"])
        a = int(timing["arti_steps"])
        protocol = timing.get("protocol", "ketchup" if spec.get("object") == "ketchup" else "standard")
        if protocol not in ("standard", "ketchup") or min(g, t) < 1 or a <= 100:
            raise ValueError("Invalid protocol timing (arti_steps must exceed 100)")
        total = g + (2 if protocol == "ketchup" else 3) * t + a
        if len(states) != total + 1:
            raise ValueError("Expected %d state rows, got %d" % (total + 1, len(states)))
        for row in states:
            if len(row) != GLOBAL_STATE_SIZE:
                raise ValueError("Expected a 138-field compose_eval global state")
            if not all(math.isfinite(float(value)) for value in row):
                raise ValueError("Nonfinite value in rollout")

        final = states[-1]
        target_position = tuple(float(v) for v in spec.get("target_position_m", final[TARGET_POSITION]))
        target_quaternion = spec.get("target_quaternion_wxyz", final[TARGET_QUATERNION])
        target_angle = float(spec.get("target_articulation_rad", final[TARGET_ANGLE]))
        initial_angle = float(spec.get("initial_articulation_rad", states[0][OBJECT_ANGLE]))
        target_y = float(spec.get("target_euler_y_rad", public_euler_y_rad(target_quaternion)))
        if len(target_position) != 3 or not all(math.isfinite(v) for v in target_position + (target_angle, initial_angle, target_y)):
            raise ValueError("Invalid target specification")
        pe = math.sqrt(sum((float(x) - y) ** 2 for x, y in zip(final[OBJECT_POSITION], target_position)))
        ae = quaternion_geodesic_rad(final[OBJECT_QUATERNION], target_quaternion)
        aae = abs(float(final[OBJECT_ANGLE]) - target_angle)

        # Original Python reads global_state at the start of each loop, before
        # refreshing it with observe/get_global_state and then calling step.
        def legacy_state(step):
            return states[max(0, step - 1)]

        arrive_start, arrive_end = g + t, g + 2 * t
        arrived = any(float(legacy_state(k)[0]) < 0.05 for k in range(arrive_start, arrive_end))
        ori_threshold = 0.15 if protocol == "ketchup" else 0.2
        # The public scripts use [-5:-1], i.e. the TARGET quaternion, not the
        # physical quaternion at [125:129]. Preserve this only for compatibility.
        public_orientation = [abs(target_y - public_euler_y_rad(legacy_state(k)[TARGET_QUATERNION]))
                              for k in range(arrive_start, arrive_end)]
        ori_get = any(error < ori_threshold for error in public_orientation)
        arti_start = g + (2 if protocol == "ketchup" else 3) * t + 100
        if protocol == "ketchup":
            opened = any(abs(float(legacy_state(k)[3]) - initial_angle) > 0.3
                         for k in range(arti_start, total))
            # This assignment happens on every step of the 100-step preparation.
            pe_public = float(legacy_state(arrive_end + 99)[0])
        else:
            opened = any(abs(float(legacy_state(k)[3]) - target_angle) < 0.5
                         for k in range(arti_start, total))
            pe_public = float(legacy_state(arrive_end)[0])
        ae_public = public_orientation[-1]
        terminal_public = states[-2]
        aae_public = abs(float(terminal_public[3]) - target_angle)
        guard = float(terminal_public[0]) < (0.4 if protocol == "ketchup" else 1.0)
        if protocol == "standard":
            guard = guard and float(terminal_public[3]) < 2.0

        paper_position_success = pe < PAPER_POSITION_THRESHOLD_M
        paper_orientation_success = ae < PAPER_ORIENTATION_THRESHOLD_RAD
        paper_articulation_success = aae < PAPER_ARTICULATION_THRESHOLD_RAD
        return {
            "valid": True,
            "invalid_reason": None,
            "paper_success": bool(paper_position_success and paper_orientation_success and paper_articulation_success),
            "paper_position_success": bool(paper_position_success),
            "paper_orientation_success": bool(paper_orientation_success),
            "paper_articulation_success": bool(paper_articulation_success),
            "paper_position_error_m": pe,
            "paper_orientation_error_rad": ae,
            "paper_articulation_error_rad": aae,
            "public_success": bool(arrived and ori_get and opened and guard),
            "public_arrived": bool(arrived),
            "public_orientation_reached": bool(ori_get),
            "public_opened": bool(opened),
            "public_terminal_guard": bool(guard),
            "public_position_error_m": pe_public,
            "public_orientation_error_rad": ae_public,
            "public_articulation_error_rad": aae_public,
            "public_position_error_clipped_m": min(0.1, max(0.0, pe_public)),
            "public_orientation_error_clipped_rad": min(0.3, max(0.0, ae_public)),
            "public_articulation_error_clipped_rad": min(1.0, max(0.0, aae_public)),
        }
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as error:
        return _invalid(error)


def evaluate_goal_diagnostics(states, spec, timing, valid=True):
    """Measure the requested intermediate pose and a stricter hinge tolerance.

    These are additional experimental metrics. The paper and public metrics in
    evaluate_episode remain unchanged. Lift position and orientation must be
    reached simultaneously, so independently visiting each cannot pass.
    """
    errors = ("lift_position_error_min_m", "lift_orientation_error_min_rad",
              "lift_position_error_final_m", "lift_orientation_error_final_rad",
              "initial_articulation_error_rad", "articulation_motion_rad",
              "articulation_start_angle_rad", "articulation_start_error_rad", "articulation_phase_motion_rad",
              "requested_final_rotation_rad", "requested_lift_rotation_rad")
    result = {name: None for name in errors}
    result.update(diagnostic_valid=False, lift_pose_success=False,
                  articulation_strict_success=False, strict_goal_success=False,
                  articulation_initial_within_paper_tolerance=None,
                  articulation_initial_within_strict_tolerance=None,
                  articulation_start_within_paper_tolerance=None,
                  articulation_start_within_strict_tolerance=None, ood_goal_success=False)
    if not valid:
        return result
    try:
        g, t = int(timing["grasp_steps"]), int(timing["transit_steps"])
        final = states[-1]
        position = spec.get("target_position_m", final[TARGET_POSITION])
        quaternion = spec.get("target_quaternion_wxyz", final[TARGET_QUATERNION])
        angle = float(spec.get("target_articulation_rad", final[TARGET_ANGLE]))
        initial_angle = float(spec.get("initial_articulation_rad", states[0][OBJECT_ANGLE]))
        lift = spec.get("lift_target_position_m")
        lift_q = spec.get("lift_target_quaternion_wxyz", quaternion)
        height = float(spec["target"]["lift_m"])
        if lift is None:
            lift = [float(position[0]), float(position[1]), float(states[0][124]) + height]
        pe = math.sqrt(sum((float(value) - float(goal)) ** 2
                           for value, goal in zip(final[OBJECT_POSITION], position)))
        ae = quaternion_geodesic_rad(final[OBJECT_QUATERNION], quaternion)
        aae = abs(float(final[OBJECT_ANGLE]) - angle)
        lift_position, lift_orientation = [], []
        height_reached = False
        for state in states[g + 1:g + t + 1]:
            lift_position.append(math.sqrt(sum((float(value) - float(goal)) ** 2
                                                for value, goal in zip(state[OBJECT_POSITION], lift))))
            lift_orientation.append(quaternion_geodesic_rad(state[OBJECT_QUATERNION], lift_q))
            height_reached |= abs(float(state[124]) - float(states[0][124]) - height) < 0.05
        lift_success = any(p < 0.05 and a < 0.2 for p, a in zip(lift_position, lift_orientation))
        intermediate = bool(height_reached and (lift_success if spec.get("require_lift_pose_success") else True))
        initial_error = abs(initial_angle - angle)
        protocol = timing.get("protocol", "ketchup" if spec.get("object") == "ketchup" else "standard")
        start_step = g + (2 if protocol == "ketchup" else 3) * t + 100
        start_angle = float(states[start_step][OBJECT_ANGLE])
        start_error = abs(start_angle - angle)
        result.update({
            "diagnostic_valid": True,
            "lift_pose_success": bool(lift_success),
            "lift_position_error_min_m": min(lift_position),
            "lift_orientation_error_min_rad": min(lift_orientation),
            "lift_position_error_final_m": lift_position[-1],
            "lift_orientation_error_final_rad": lift_orientation[-1],
            "initial_articulation_error_rad": initial_error,
            "articulation_initial_within_paper_tolerance": initial_error < 0.5,
            "articulation_initial_within_strict_tolerance": initial_error < 0.1,
            "articulation_motion_rad": abs(float(final[OBJECT_ANGLE]) - initial_angle),
            "articulation_start_angle_rad": start_angle,
            "articulation_start_error_rad": start_error,
            "articulation_start_within_paper_tolerance": start_error < 0.5,
            "articulation_start_within_strict_tolerance": start_error < 0.1,
            "articulation_phase_motion_rad": abs(float(final[OBJECT_ANGLE]) - start_angle),
            "articulation_strict_success": aae < 0.1,
            "requested_final_rotation_rad": quaternion_geodesic_rad(states[0][OBJECT_QUATERNION], quaternion),
            "requested_lift_rotation_rad": quaternion_geodesic_rad(states[0][OBJECT_QUATERNION], lift_q),
            "ood_goal_success": bool(pe < 0.05 and ae < 0.2 and aae < 0.5 and intermediate),
            "strict_goal_success": bool(pe < 0.05 and ae < 0.2 and aae < 0.1 and intermediate),
        })
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as error:
        result["diagnostic_error"] = str(error)
    return result

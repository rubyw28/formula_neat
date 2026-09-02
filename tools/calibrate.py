"""Report what the current physics constants mean on a real track.

    python tools/calibrate.py oval monza

`car.py` decides a crash from centrifugal force against available grip, which
is hard to reason about directly. This mirrors that maths and answers the
questions that actually matter when tuning `sim_config.py`:

  * how fast can a car get through the tightest corner,
  * can it physically steer that tightly at that speed,
  * and how many ticks does a clean lap take, which sets the per-generation
    tick budget in `main.py`.

Nothing here is imported by the simulation; it is a tuning aid.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_config import SIM_CFG  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASS_KG = 768.0      # matches car.py
WHEELBASE_M = 3.4    # matches car.py


def steer_sensitivity(cfg, speed, aero):
    value = max(0.45, min(1.0, 1.0 - 0.40 * (speed / cfg["max_speed_base"])))
    return value * (0.85 if aero == "X" else 1.0)


def centrifugal_force(cfg, speed, steering_delta_deg):
    """Force car.py would compute for this speed and this much steering."""
    wheel = min(
        abs(steering_delta_deg) * cfg["steer_to_wheel_angle"],
        cfg["max_wheel_angle_deg"],
    )
    if math.radians(wheel) < 0.01:
        return 0.0
    radius_m = WHEELBASE_M / math.tan(math.radians(wheel))
    speed_mps = speed * cfg["pixel_to_meter"] * cfg["fps"]
    return MASS_KG * speed_mps * speed_mps / radius_m


def grip_limit(cfg, grip, aero):
    mult = cfg["z_mode_grip_mult"] if aero == "Z" else cfg["x_mode_grip_mult"]
    return cfg["base_grip_limit_n"] * grip * mult


def corner_speed(cfg, path_radius_px, grip=1.0, aero="Z"):
    """Fastest the car can hold a corner of this radius, in px/tick.

    Limited by grip, and separately by how far the steering can actually move
    at that speed. Returns (speed, which limit bit first).
    """
    top = cfg["max_speed_base"] + (cfg["x_mode_speed_bonus"] if aero == "X" else 0.0)
    limit = grip_limit(cfg, grip, aero)

    best, reason = cfg["min_speed"], "grip"
    speed = cfg["min_speed"]
    while speed <= top:
        needed = math.degrees(speed / path_radius_px)
        steerable = cfg["steer_max_per_tick"] * steer_sensitivity(cfg, speed, aero)
        if needed > steerable:
            return best, "steering lock"
        if centrifugal_force(cfg, speed, needed) > limit:
            return best, "grip"
        best, reason = speed, "flat out"
        speed += 0.05
    return top, reason


def smooth_line(line, window=5):
    """Moving average around the closed line.

    Racing lines are stored as whole pixels, and that rounding alone is enough
    to fake corners far tighter than the track has.
    """
    n = len(line)
    half = window // 2
    out = []
    for i in range(n):
        xs = [line[(i + k) % n][0] for k in range(-half, half + 1)]
        ys = [line[(i + k) % n][1] for k in range(-half, half + 1)]
        out.append((sum(xs) / window, sum(ys) / window))
    return out


def path_radii(line, stencil=4):
    """Radius of curvature at each racing-line point, in pixels.

    Fitted as the circumradius of the point and its neighbours `stencil` steps
    either side. A tighter stencil just measures rounding noise in the line and
    reports corners far sharper than the track really has.
    """
    radii = []
    n = len(line)
    for i in range(n):
        p0 = line[(i - stencil) % n]
        p1 = line[i]
        p2 = line[(i + stencil) % n]
        a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        c = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        area2 = abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        )
        radii.append(1e9 if area2 < 1e-9 else (a * b * c) / (2.0 * area2))
    return radii


def report(cfg, name):
    with open(os.path.join(ROOT, "assets", "tracks", f"{name}.json")) as handle:
        meta = json.load(handle)
    line = smooth_line(meta["racing_line"])
    radii = sorted_radii = path_radii(line)
    spacing = meta["length_px"] / len(line)

    print(f"\n=== {name} ===")
    print(f"  lap {meta['length_px']}px, width ~{meta['median_width_px']}px, "
          f"{len(line)} racing-line points ({spacing:.0f}px apart)")

    top = cfg["max_speed_base"]
    top_x = top + cfg["x_mode_speed_bonus"]
    to_kmh = cfg["pixel_to_meter"] * cfg["fps"] * 3.6
    print(f"  top speed  Z {top:.1f}px/tick ({top * to_kmh:.0f} km/h)   "
          f"X {top_x:.1f}px/tick ({top_x * to_kmh:.0f} km/h)")

    ranked = sorted(sorted_radii)
    # 5th percentile rather than the outright minimum: one stray point should
    # not define the whole track's grip budget.
    tightest = ranked[max(0, int(len(ranked) * 0.05))]
    print(f"  corner radius: tightest {ranked[0]:.0f}px, "
          f"5th pct {tightest:.0f}px, median {ranked[len(ranked) // 2]:.0f}px")
    for label, grip, aero in (
        ("fresh Z", 1.0, "Z"),
        ("fresh X", 1.0, "X"),
        ("worn  Z", 0.45, "Z"),
    ):
        speed, why = corner_speed(cfg, tightest, grip, aero)
        print(f"  tightest corner (r={tightest:.0f}px) on {label}: "
              f"{speed:.2f}px/tick ({speed * to_kmh:.0f} km/h) - limited by {why}")

    ticks = sum(
        spacing / max(corner_speed(cfg, r)[0], 0.1) for r in radii
    )
    print(f"  clean lap ~{ticks:.0f} ticks ({ticks / cfg['fps']:.1f}s at "
          f"{cfg['fps']}fps), slowest corner {min(corner_speed(cfg, r)[0] for r in radii):.1f}px/tick")
    return ticks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", nargs="*", default=None)
    args = parser.parse_args()

    names = args.tracks or ["oval", "monza"]
    worst = max(report(SIM_CFG, name) for name in names)
    print(f"\nSuggested per-generation budget: >= {int(worst * 1.6)} ticks "
          f"to give a competent car time for a lap and a bit.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

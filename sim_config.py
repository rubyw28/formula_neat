"""Physics, power unit and sensor constants for the Formula NEAT simulation.

`car.py` reads every one of these through `self.cfg`. They are gathered here so
the car model stays free of magic numbers and so a run can override any of them
without editing code.

Units
-----
Speed is pixels per tick; `pixel_to_meter` and `fps` convert it to m/s for the
force calculations, so the grip model is in real newtons on a real 768 kg car.
Battery energy is a 0..1 fraction of a full store.

Cornering
---------
`car.py` estimates a turn radius from the steering angle with a bicycle model
and crashes the car when the resulting centrifugal force exceeds available grip.
The practical effect is a speed/steering envelope: the faster you go, the less
you may turn. `tools/calibrate.py` prints that envelope for the current numbers.
"""

import copy

SIM_CFG = {
    # --- rendering / geometry -------------------------------------------
    "screen_width": 1920,
    "screen_height": 1080,
    "fps": 60,
    "border_color": (255, 255, 255, 255),
    "car_sprite_path": "assets/cars/car.png",
    "car_size_x": 34,
    "car_size_y": 16,
    # One pixel is 9 cm, so the 1920px frame is a ~173 m wide circuit.
    "pixel_to_meter": 0.09,

    # --- sensors --------------------------------------------------------
    # Radars fire from the nose rather than the middle of the car, so the
    # network sees what it is about to hit instead of what surrounds it.
    "sensor_nose_offset": 17,
    "radar_max_distance": 260,
    # Radars stride this far per probe, then walk the last stride pixel by
    # pixel. 1 disables striding.
    "radar_coarse_step": 4,
    "radar_angles": [-90, -45, 0, 45, 90],

    # --- tyres ----------------------------------------------------------
    "grip_base": 1.0,
    # Tuned so a car loses roughly half its grip over a long stint.
    "grip_wear_rate": 0.00015,

    # --- steering -------------------------------------------------------
    "steer_max_per_tick": 4.0,
    "steer_to_wheel_angle": 6.0,
    "max_wheel_angle_deg": 22.0,

    # --- grip limit -----------------------------------------------------
    # ~4 g of lateral grip on fresh tyres in Z mode. X mode trades downforce
    # for straight-line speed, so it corners meaningfully worse.
    "base_grip_limit_n": 30000.0,
    "z_mode_grip_mult": 1.0,
    "x_mode_grip_mult": 0.72,

    # --- power unit -----------------------------------------------------
    "engine_accel": 0.16,
    "drag_coeff": 0.0022,
    "brake_decel": 0.45,
    "max_speed_base": 9.0,
    "min_speed": 1.5,
    "x_mode_speed_bonus": 1.8,

    # --- battery (0..1) -------------------------------------------------
    # A full store is worth about five seconds of maximum deployment.
    "battery_deploy_rate": 0.0035,
    "battery_harvest_brake": 0.0060,
    "battery_harvest_lift": 0.0012,
    "battery_override_drain": 0.0100,

    # --- fitness --------------------------------------------------------
    # Progress has to out-earn presence. With the original weights a car that
    # simply sat on the racing line at minimum speed banked 2.08 per tick -
    # 2495 over a generation - while a whole lap of the oval paid 580. The
    # optimum was to crawl, and that is exactly what training converged on.
    # Per-tick terms are now small enough that a car has to actually go
    # somewhere to score.
    "reward_checkpoint": 18.0,
    "reward_lap": 220.0,
    "reward_apex": 2.2,
    # Racing-line reward: full value on the line, zero `falloff` px away from
    # it, negative beyond that.
    "reward_line_max": 0.30,
    "reward_line_falloff": 55.0,
    "reward_pace": 0.10,
    "reward_battery_ok": 0.05,
    "penalty_battery_low": 0.35,
    "penalty_time": 0.30,
    "penalty_steering": 0.01,

    # --- manual override ------------------------------------------------
    "override_distance_px": 120,
    "override_boost": 0.09,
    "override_speed_bonus": 1.6,
    # Deliberately 0. car.py stamps `last_override_tick` on every tick the
    # override is active, so any non-zero cooldown makes the override fire for a
    # single tick and then lock itself out. Battery drain is the real limiter.
    "override_cooldown_ticks": 0,
}


def make_cfg(**overrides):
    """A copy of the defaults with individual constants replaced."""
    cfg = copy.deepcopy(SIM_CFG)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown sim config keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg

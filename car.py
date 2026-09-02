import math

import pygame


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def distance_xy(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


_SPRITE_CACHE = {}


def load_sprite(path, size):
    """Load and scale the car sprite once per (file, size).

    Every car used to re-read and re-scale the PNG from disk, for every car in
    every generation. smoothscale rather than scale because the source art is
    1059x476 and the car is a few dozen pixels: point-sampling a 30:1 reduction
    throws away most of the car.
    """
    key = (path, size)
    sprite = _SPRITE_CACHE.get(key)
    if sprite is None:
        sprite = pygame.image.load(path).convert_alpha()
        sprite = pygame.transform.smoothscale(sprite, size)
        _SPRITE_CACHE[key] = sprite
    return sprite


class Car:
    def __init__(self, spawn_x, spawn_y, spawn_angle, track_meta, sim_cfg):
        self.cfg = sim_cfg
        self.sprite = load_sprite(
            self.cfg["car_sprite_path"],
            (self.cfg["car_size_x"], self.cfg["car_size_y"]),
        )
        self.rotated_sprite = self.sprite

        self.position = [float(spawn_x), float(spawn_y)]
        self.angle = float(spawn_angle)
        self.speed = 0.0
        self.center = [
            self.position[0] + self.cfg["car_size_x"] / 2,
            self.position[1] + self.cfg["car_size_y"] / 2,
        ]
        self.sensor_origin = list(self.center)
        self.radars = []
        self.alive = True

        # 2026 chassis/power-unit state.
        self.mass_kg = 768.0
        self.wheelbase_m = 3.4
        self.chassis_width_m = 1.9
        self.aero_mode = "Z"  # Z-mode default for cornering downforce.
        self.battery_energy = 1.0
        self.override_active = False
        self.override_allowed = False
        self.last_override_tick = -10_000

        self.distance = 0.0
        self.time = 0
        self.lap_ticks = 0
        self.completed_laps = 0
        self.last_control = {
            "steer": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "aero": 0.0,
            "override": 0.0,
        }
        self.grip = self.cfg["grip_base"]
        self.tire_wear = 0.0
        self.g_force = 0.0
        self.corner_load_ratio = 0.0

        self.checkpoints = track_meta.get("checkpoints", [])
        self.next_checkpoint = 0
        self.last_checkpoint_tick = 0
        self.racing_line = track_meta.get("racing_line", [])
        self.apex_indices = set(track_meta.get("apex_indices", []))

        self.reward_delta = 0.0
        # Cursor into racing_line, so the nearest-point search can start local.
        self._line_index = -1
        self._line_search_span = 14

    def sprite_rect(self):
        """Where the rotated sprite sits on screen, centred on the car."""
        rect = self.rotated_sprite.get_rect()
        rect.center = (int(self.center[0]), int(self.center[1]))
        return rect

    def draw(self, screen, debug_draw):
        screen.blit(self.rotated_sprite, self.sprite_rect().topleft)
        if not debug_draw:
            return
        # Outline what actually collides: the rotated sprite's rectangle. The
        # unrotated car_size_x by car_size_y box shares a centre with it but is
        # the wrong shape as soon as the car turns, so it never matched the
        # footprint the collision mask uses.
        pygame.draw.rect(screen, (30, 144, 255), self.sprite_rect(), 1)
        for radar in self.radars:
            end_pos = radar[0]
            start_pos = (int(self.sensor_origin[0]), int(self.sensor_origin[1]))
            pygame.draw.line(screen, (0, 255, 0), start_pos, end_pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), end_pos, 3)
        pygame.draw.circle(
            screen,
            (255, 0, 0),
            (int(self.sensor_origin[0]), int(self.sensor_origin[1])),
            4,
        )

    def update_sensor_origin(self):
        nose_dx = (
            math.cos(math.radians(360 - self.angle))
            * self.cfg["sensor_nose_offset"]
        )
        nose_dy = (
            math.sin(math.radians(360 - self.angle))
            * self.cfg["sensor_nose_offset"]
        )
        self.sensor_origin = [self.center[0] + nose_dx, self.center[1] + nose_dy]

    def rotate_center(self, image, angle):
        # Keep the whole rotated image. Cropping it back to the unrotated rect
        # is impossible past ~45deg - a rotated 34x16 car is only 16px wide at
        # 90deg - and it would clip the nose and tail off the collision mask.
        # sprite_rect() is what re-centres it on the car.
        return pygame.transform.rotate(image, angle)

    def check_collision(self, border_mask):
        car_mask = pygame.mask.from_surface(self.rotated_sprite)
        offset = self.sprite_rect().topleft
        out_of_bounds = (
            self.position[0] < 0
            or self.position[1] < 0
            or self.position[0] + self.cfg["car_size_x"] >= self.cfg["screen_width"]
            or self.position[1] + self.cfg["car_size_y"] >= self.cfg["screen_height"]
        )
        self.alive = not out_of_bounds and border_mask.overlap(car_mask, offset) is None

    def check_radar(self, degree, game_map):
        """Cast one radar from the nose and record where it meets a wall.

        Marched, not walked pixel by pixel: the track's clearance field says how
        far the nearest wall is from any point, so the ray can safely jump that
        far in one go. Open track is crossed in a handful of steps and the
        result is still exact, because within the clearance there is by
        definition nothing to hit.

        Without a clearance field it falls back to fixed steps of
        `radar_coarse_step`, which is only safe when that is 1 - a larger stride
        can straddle a wall at a glancing angle and report the far side of it.
        """
        heading = math.radians(360 - (self.angle + degree))
        step_x = math.cos(heading)
        step_y = math.sin(heading)
        origin_x, origin_y = self.sensor_origin

        max_distance = self.cfg["radar_max_distance"]
        width = self.cfg["screen_width"]
        height = self.cfg["screen_height"]
        field = self.cfg.get("clearance_field")
        border = self.cfg["border_color"]
        fallback_step = max(1, self.cfg["radar_coarse_step"])

        length = 0.0
        x, y = int(origin_x), int(origin_y)
        while length < max_distance:
            x = int(origin_x + step_x * length)
            y = int(origin_y + step_y * length)
            if x < 0 or y < 0 or x >= width or y >= height:
                break
            if field is not None:
                # 4-connected BFS overestimates true distance by up to sqrt(2),
                # so scale it down before trusting it as a safe jump.
                clear = field.get_at((x, y))[0]
                if clear == 0:
                    break
                # Half-pixel steps once we are alongside a wall. A ray running
                # nearly parallel to the edge skims a one-pixel-wide strip, and
                # at whole-pixel steps whether it clips the wall comes down to
                # rounding; sub-sampling there costs almost nothing because it
                # only happens in the last few pixels of a ray.
                length += max(0.5, clear / 1.45)
            else:
                if game_map.get_at((x, y)) == border:
                    break
                length += fallback_step

        dist = int(math.hypot(x - origin_x, y - origin_y))
        self.radars.append([(x, y), dist])

    def _nearest_racing_line_distance(self):
        if not self.racing_line:
            return 0.0, -1

        count = len(self.racing_line)
        # The car only moves a few pixels a tick, so the nearest racing-line
        # point is next to last tick's. Scanning the whole line every tick for
        # every car was the second most expensive thing here.
        if self._line_index < 0:
            candidates = range(count)
        else:
            span = self._line_search_span
            candidates = [
                (self._line_index + offset) % count
                for offset in range(-span, span + 1)
            ]

        best_dist = 1e9
        best_idx = -1
        best_offset = 0
        for position, i in enumerate(candidates):
            d = distance_xy(self.center, self.racing_line[i])
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_offset = position

        # If the winner sits on the edge of the window we may have missed a
        # closer point outside it, so fall back to the full sweep.
        if self._line_index >= 0 and best_offset in (0, len(candidates) - 1):
            best_dist = 1e9
            best_idx = -1
            for i, point in enumerate(self.racing_line):
                d = distance_xy(self.center, point)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        self._line_index = best_idx
        return best_dist, best_idx

    def _update_checkpoint_progress(self):
        if not self.checkpoints:
            return 0.0
        cp = self.checkpoints[self.next_checkpoint]
        cp_radius = cp.get("radius", 30)
        cp_dist = distance_xy(self.center, (cp["x"], cp["y"]))
        if cp_dist > cp_radius:
            return 0.0
        reward = self.cfg["reward_checkpoint"]
        self.next_checkpoint += 1
        self.last_checkpoint_tick = self.time
        if self.next_checkpoint >= len(self.checkpoints):
            self.next_checkpoint = 0
            self.completed_laps += 1
            self.lap_ticks = self.time
            reward += self.cfg["reward_lap"]
        return reward

    def _update_tire_wear(self):
        wear_gain = self.cfg["grip_wear_rate"] * (
            1.0 + self.speed / max(self.cfg["max_speed_base"], 1.0)
        )
        self.tire_wear = clamp(self.tire_wear + wear_gain, 0.0, 1.0)
        self.grip = clamp(
            self.cfg["grip_base"] * (1.0 - 0.55 * self.tire_wear), 0.35, 1.1
        )

    def _estimate_turn_radius(self, steering_delta_deg):
        steer_angle_deg = clamp(
            steering_delta_deg * self.cfg["steer_to_wheel_angle"],
            -self.cfg["max_wheel_angle_deg"],
            self.cfg["max_wheel_angle_deg"],
        )
        steer_angle_rad = math.radians(abs(steer_angle_deg))
        if steer_angle_rad < 0.01:
            return 1e9
        return self.wheelbase_m / math.tan(steer_angle_rad)

    def _compute_centrifugal_force(self, speed_px_per_tick, turn_radius_m):
        speed_mps = speed_px_per_tick * self.cfg["pixel_to_meter"] * self.cfg["fps"]
        if turn_radius_m <= 0:
            return 1e9
        return (self.mass_kg * speed_mps * speed_mps) / turn_radius_m

    def _current_grip_limit(self):
        aero_mult = (
            self.cfg["z_mode_grip_mult"]
            if self.aero_mode == "Z"
            else self.cfg["x_mode_grip_mult"]
        )
        return self.cfg["base_grip_limit_n"] * self.grip * aero_mult

    def _apply_aero_and_override(self, aero_signal, override_signal, nearest_car_distance, tick):
        self.aero_mode = "X" if aero_signal > 0.0 else "Z"
        self.override_allowed = (
            nearest_car_distance is not None
            and nearest_car_distance <= self.cfg["override_distance_px"]
        )
        override_request = override_signal > 0.2
        can_override = (
            self.override_allowed
            and self.battery_energy > 0.12
            and (tick - self.last_override_tick) >= self.cfg["override_cooldown_ticks"]
        )
        self.override_active = override_request and can_override
        if self.override_active:
            self.last_override_tick = tick

    def _apply_powertrain(self, throttle, brake):
        # 2026 simplification: half potential from ICE, half from electrical deployment.
        ice_accel = self.cfg["engine_accel"] * throttle * 0.5
        electric_available = clamp(self.battery_energy * 1.5, 0.0, 1.0)
        electric_accel = self.cfg["engine_accel"] * throttle * 0.5 * electric_available

        if self.override_active:
            electric_accel += self.cfg["override_boost"]

        drag_term = self.cfg["drag_coeff"] * self.speed * self.speed
        self.speed += ice_accel + electric_accel - (self.cfg["brake_decel"] * brake) - drag_term
        max_speed = (
            self.cfg["max_speed_base"]
            if self.aero_mode == "Z"
            else self.cfg["max_speed_base"] + self.cfg["x_mode_speed_bonus"]
        )
        if self.override_active:
            max_speed += self.cfg["override_speed_bonus"]
        self.speed = clamp(self.speed, self.cfg["min_speed"], max_speed)

        deploy_drain = (
            self.cfg["battery_deploy_rate"] * throttle * electric_available
            + (self.cfg["battery_override_drain"] if self.override_active else 0.0)
        )
        harvest_gain = (
            self.cfg["battery_harvest_brake"] * brake
            + self.cfg["battery_harvest_lift"] * max(0.0, 0.25 - throttle) * 4.0
        )
        self.battery_energy = clamp(
            self.battery_energy - deploy_drain + harvest_gain,
            0.0,
            1.0,
        )

    def _apply_cornering_crash_logic(self, steering_delta_deg, steer_input):
        turn_radius_m = self._estimate_turn_radius(steering_delta_deg)
        fc = self._compute_centrifugal_force(self.speed, turn_radius_m)
        grip_limit = self._current_grip_limit()
        self.corner_load_ratio = fc / max(grip_limit, 1.0)
        self.g_force = fc / max(self.mass_kg * 9.81, 1.0)

        # Small warmup window to prevent immediate generation-0 wipeouts.
        if self.time < 45:
            return

        # X-mode in sharp corners should be risky by design.
        if (
            self.aero_mode == "X"
            and abs(steer_input) > 0.80
            and self.speed > (self.cfg["max_speed_base"] * 0.90)
        ):
            self.alive = False
            return
        if fc > grip_limit:
            self.alive = False

    def update(self, game_map, border_mask, output, nearest_car_distance, tick):
        if not self.alive:
            return

        self.time += 1
        self._update_tire_wear()

        steer_input = clamp(output[0], -1.0, 1.0)
        throttle = clamp((output[1] + 1.0) / 2.0, 0.0, 1.0)
        brake = clamp((output[2] + 1.0) / 2.0, 0.0, 1.0)
        aero_signal = output[3]
        override_signal = output[4]
        self.last_control = {
            "steer": steer_input,
            "throttle": throttle,
            "brake": brake,
            "aero": aero_signal,
            "override": override_signal,
        }

        self._apply_aero_and_override(
            aero_signal, override_signal, nearest_car_distance, tick
        )

        steer_sensitivity = clamp(
            1.0 - (0.40 * (self.speed / max(self.cfg["max_speed_base"], 1.0))),
            0.45,
            1.0,
        )
        if self.aero_mode == "X":
            steer_sensitivity *= 0.85
        steering_delta = steer_input * self.cfg["steer_max_per_tick"] * steer_sensitivity
        self.angle += steering_delta

        self._apply_powertrain(throttle, brake)
        self._apply_cornering_crash_logic(steering_delta, steer_input)
        if not self.alive:
            return

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.distance += self.speed
        self.center = [
            int(self.position[0]) + self.cfg["car_size_x"] / 2,
            int(self.position[1]) + self.cfg["car_size_y"] / 2,
        ]
        self.update_sensor_origin()
        self.check_collision(border_mask)
        if not self.alive:
            return

        self.radars.clear()
        for degree in self.cfg["radar_angles"]:
            self.check_radar(degree, game_map)

        cfg = self.cfg
        checkpoint_reward = self._update_checkpoint_progress()
        line_dist, line_idx = self._nearest_racing_line_distance()
        line_reward = cfg["reward_line_max"] * clamp(
            1.0 - (line_dist / cfg["reward_line_falloff"]), -2.0, 1.0
        )
        apex_bonus = (
            cfg["reward_apex"]
            if line_idx in self.apex_indices and line_dist < 25
            else 0.0
        )

        # Efficient pace: reward speed with battery discipline.
        pace_reward = cfg["reward_pace"] * self.speed
        low_battery_penalty = (
            cfg["penalty_battery_low"] if self.battery_energy < 0.12 else 0.0
        )
        battery_efficiency_bonus = (
            cfg["reward_battery_ok"] if self.battery_energy > 0.35 else 0.0
        )
        time_penalty = cfg["penalty_time"]
        steering_penalty = cfg["penalty_steering"] * abs(steer_input)
        self.reward_delta = (
            checkpoint_reward
            + line_reward
            + apex_bonus
            + pace_reward
            + battery_efficiency_bonus
            - low_battery_penalty
            - steering_penalty
            - time_penalty
        )

    def get_data(self):
        radar_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            radar_values[i] = clamp(
                radar[1] / self.cfg["radar_max_distance"], 0.0, 1.0
            )

        checkpoint_dist_norm = 1.0
        checkpoint_heading_norm = 0.0
        if self.checkpoints:
            cp = self.checkpoints[self.next_checkpoint]
            cp_vec_x = cp["x"] - self.center[0]
            cp_vec_y = cp["y"] - self.center[1]
            cp_dist = math.sqrt(cp_vec_x * cp_vec_x + cp_vec_y * cp_vec_y)
            checkpoint_dist_norm = clamp(cp_dist / 600.0, 0.0, 1.0)
            cp_heading = math.degrees(math.atan2(cp_vec_y, cp_vec_x))
            heading_error = (cp_heading - (360 - self.angle) + 540) % 360 - 180
            checkpoint_heading_norm = heading_error / 180.0

        aero_flag = 1.0 if self.aero_mode == "X" else 0.0
        return radar_values + [
            clamp(
                self.speed
                / (
                    self.cfg["max_speed_base"]
                    + self.cfg["x_mode_speed_bonus"]
                    + self.cfg["override_speed_bonus"]
                ),
                0.0,
                1.0,
            ),
            self.grip,
            self.battery_energy,
            aero_flag,
            1.0 if self.override_allowed else 0.0,
            clamp(self.corner_load_ratio, 0.0, 2.0),
            checkpoint_dist_norm,
            checkpoint_heading_norm,
        ]

    def get_reward(self):
        return self.reward_delta

    def is_alive(self):
        return self.alive

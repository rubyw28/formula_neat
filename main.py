"""Formula NEAT - train cars to race with simplified 2026 F1 mechanics.

    python main.py                          # train on the oval, with a window
    python main.py --track monza            # train on Monza
    python main.py --headless -g 200        # train fast, no rendering
    python main.py --replay best_oval.pkl   # watch a saved car drive

This is the runner for the physics model in `car.py`: grip and centrifugal
force, tyre wear, active aero, battery deployment and manual override. The old
discrete-action baseline still lives in `newcar.py`, frozen apart from now
reading config_baseline.txt.

Controls while a window is open:
    Esc     quit                D   debug overlay (radars, racing line)
    Space   end this generation H   hide the telemetry panel
    1-5     run 1-5 sim ticks per drawn frame
"""

import argparse
import math
import os
import pickle
import sys
import time

from car import Car
from sim_config import make_cfg
from track import Track

ROOT = os.path.dirname(os.path.abspath(__file__))

PANEL_BG = (16, 18, 22, 215)
INK = (236, 238, 242)
DIM = (150, 156, 168)
ACCENT = (0, 200, 255)
WARN = (255, 172, 40)
GOOD = (90, 220, 130)


# --------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--track", default="oval", help="track name (default: oval)")
    parser.add_argument("--config", default="config.txt", help="NEAT config file")
    parser.add_argument(
        "-g", "--generations", type=int, default=1000, help="generations to run"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=1200,
        help="simulation ticks per generation (default: 1200, about two laps)",
    )
    parser.add_argument(
        "--stall-ticks",
        type=int,
        default=260,
        help="retire a car that has not reached a checkpoint in this long",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="frame cap for rendering only; the physics tick rate is fixed in "
        "sim_config.py",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="window scale (default: fit the desktop)",
    )
    parser.add_argument("--headless", action="store_true", help="train with no window")
    parser.add_argument("--debug", action="store_true", help="start with overlays on")
    parser.add_argument(
        "--save-best",
        default=None,
        help="write the best genome here whenever it improves "
        "(default: best_<track>.pkl)",
    )
    parser.add_argument("--replay", default=None, help="watch a saved genome instead")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args(argv)


# What car.py's sensor and action layers actually are.
INPUTS = 13   # 5 radars + speed, grip, battery, aero, override, load, 2x checkpoint
OUTPUTS = 5   # steer, throttle, brake, aero, override


def load_neat_config(path):
    import neat

    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"NEAT config not found: {full}")
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        full,
    )
    inputs = config.genome_config.num_inputs
    outputs = config.genome_config.num_outputs
    if (inputs, outputs) != (INPUTS, OUTPUTS):
        raise SystemExit(
            f"{os.path.basename(full)} declares {inputs} inputs / {outputs} outputs, "
            f"but car.py needs {INPUTS} / {OUTPUTS}.\n"
            f"(newcar.py is the 5-input baseline; it uses config_baseline.txt.)"
        )
    return config


# --------------------------------------------------------------------------


class Simulation:
    """Runs generations of cars around a track and scores them."""

    def __init__(self, args, cfg, track, screen, world, fonts):
        self.args = args
        self.cfg = cfg
        self.track = track
        self.screen = screen
        self.world = world
        self.fonts = fonts

        self.generation = 0
        self.best_fitness = float("-inf")
        self.best_laps = 0
        self.show_hud = True
        self.debug = args.debug
        self.ticks_per_frame = 1
        self.started = time.time()
        # "GEN" while training, "RUN" while replaying a saved genome.
        self.round_label = "GEN"

    # -- lifecycle ------------------------------------------------------

    def evaluate(self, genomes, neat_config):
        """NEAT fitness function: one generation of racing."""
        import neat
        import pygame

        self.generation += 1
        cfg = self.cfg
        slots = self.track.grid_slots(len(genomes), cfg)
        meta = self.track.car_meta()

        nets, cars = [], []
        for index, (_, genome) in enumerate(genomes):
            genome.fitness = 0.0
            nets.append(neat.nn.FeedForwardNetwork.create(genome, neat_config))
            x, y, angle = slots[index]
            cars.append(Car(x, y, angle, meta, cfg))

        clock = pygame.time.Clock() if not self.args.headless else None
        skip = False

        for tick in range(self.args.ticks):
            if not self.args.headless:
                skip = self.handle_events()
                if skip:
                    break

            alive = self._step(tick, nets, cars, genomes)
            if alive == 0:
                break

            if not self.args.headless and tick % self.ticks_per_frame == 0:
                self.render(cars, tick, alive, len(genomes))
                clock.tick(self.args.fps)

        self._finish_generation(genomes, cars)

    def _step(self, tick, nets, cars, genomes):
        """Advance every living car by one tick. Returns how many survived."""
        # Positions are sampled once, before anything moves, so every car sees
        # the same snapshot of the field rather than a half-updated one.
        rivals = [(i, car.center) for i, car in enumerate(cars) if car.is_alive()]

        alive = 0
        for index, car in enumerate(cars):
            if not car.is_alive():
                continue

            nearest = self._nearest_rival(index, car.center, rivals)
            output = nets[index].activate(car.get_data())
            car.update(self.track.surface, self.track.border_mask, output, nearest, tick)

            if not car.is_alive():
                continue
            # A car that stops making progress is retired rather than left to
            # idle out the generation collecting racing-line reward.
            if car.time - car.last_checkpoint_tick > self.args.stall_ticks:
                car.alive = False
                continue

            genomes[index][1].fitness += car.get_reward()
            alive += 1
        return alive

    @staticmethod
    def _nearest_rival(index, centre, rivals):
        """Distance to the closest other car, or None if it is alone."""
        best = None
        for other_index, other in rivals:
            if other_index == index:
                continue
            distance = math.hypot(centre[0] - other[0], centre[1] - other[1])
            if best is None or distance < best:
                best = distance
        return best

    def _finish_generation(self, genomes, cars):
        best = max(genomes, key=lambda entry: entry[1].fitness or 0.0)
        laps = max((car.completed_laps for car in cars), default=0)
        self.best_laps = max(self.best_laps, laps)
        total_cps = len(self.track.meta["checkpoints"])
        progress = max(
            (car.completed_laps * total_cps + car.next_checkpoint for car in cars),
            default=0,
        )

        fitness = best[1].fitness or 0.0
        improved = fitness > self.best_fitness
        if improved:
            self.best_fitness = fitness
            if self.args.save_best:
                with open(self.args.save_best, "wb") as handle:
                    pickle.dump(best[1], handle)

        print(
            f"  gen {self.generation:>4} | best {fitness:9.1f} | "
            f"laps {laps} (best {self.best_laps}) | "
            f"furthest {progress} checkpoints "
            f"({progress / total_cps:.2f} laps) "
            f"{'| saved' if improved and self.args.save_best else ''}"
        )

    # -- input ----------------------------------------------------------

    def handle_events(self):
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                raise SystemExit(0)
            if event.key == pygame.K_SPACE:
                return True
            if event.key == pygame.K_d:
                self.debug = not self.debug
            if event.key == pygame.K_h:
                self.show_hud = not self.show_hud
            if pygame.K_1 <= event.key <= pygame.K_5:
                self.ticks_per_frame = event.key - pygame.K_0
        return False

    # -- rendering ------------------------------------------------------

    def render(self, cars, tick, alive, total):
        import pygame

        world = self.world
        world.blit(self.track.surface, (0, 0))

        if self.debug:
            self._draw_track_debug(world)
        for car in cars:
            if car.is_alive():
                car.draw(world, self.debug)

        leader = self._leader(cars)
        if self.show_hud:
            self._draw_hud(world, leader, tick, alive, total)

        if world is self.screen:
            pygame.display.flip()
        else:
            pygame.transform.scale(world, self.screen.get_size(), self.screen)
            pygame.display.flip()

    def _draw_track_debug(self, surface):
        import pygame

        line = self.track.racing_line
        for i in range(len(line)):
            pygame.draw.line(
                surface, (90, 90, 90), line[i], line[(i + 1) % len(line)], 2
            )
        for cp in self.track.meta["checkpoints"]:
            pygame.draw.circle(surface, (0, 90, 120), (cp["x"], cp["y"]), cp["radius"], 1)

    @staticmethod
    def _leader(cars):
        best = None
        for car in cars:
            if not car.is_alive():
                continue
            key = (car.completed_laps, car.next_checkpoint, car.distance)
            if best is None or key > best[0]:
                best = (key, car)
        return best[1] if best else None

    def _draw_hud(self, surface, leader, tick, alive, total):
        import pygame

        small, body, big = self.fonts
        width, height = 330, 300
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)

        def text(font, value, x, y, colour=INK):
            panel.blit(font.render(value, True, colour), (x, y))

        text(big, f"{self.round_label} {self.generation}", 16, 12)
        text(small, self.track.name.upper(), 16, 46, ACCENT)
        text(small, f"{alive}/{total} running", 180, 46, DIM)

        elapsed = tick / self.cfg["fps"]
        text(body, f"tick {tick}/{self.args.ticks}   {elapsed:5.1f}s", 16, 70, DIM)
        text(body, f"best fitness  {self._fmt(self.best_fitness)}", 16, 92)
        text(body, f"best laps     {self.best_laps}", 16, 112)
        wall = int(time.time() - self.started)
        text(body, f"{wall // 60}m{wall % 60:02d}s", 236, 112, DIM)

        pygame.draw.line(panel, (60, 66, 78), (16, 136), (width - 16, 136))
        text(small, "LEADER", 16, 144, ACCENT)

        if leader is None:
            text(body, "no cars running", 16, 172, DIM)
        else:
            kmh = leader.speed * self.cfg["pixel_to_meter"] * self.cfg["fps"] * 3.6
            mode = leader.aero_mode
            text(big, f"{kmh:4.0f}", 16, 166)
            text(small, "km/h", 88, 190, DIM)

            text(body, f"AERO {mode}", 150, 168, WARN if mode == "X" else GOOD)
            override = (
                "OVERRIDE ON"
                if leader.override_active
                else ("override ready" if leader.override_allowed else "override off")
            )
            text(
                small,
                override,
                150,
                190,
                GOOD if leader.override_active else (WARN if leader.override_allowed else DIM),
            )

            self._bar(panel, 16, 216, width - 32, "BATTERY", leader.battery_energy, ACCENT)
            self._bar(panel, 16, 246, width - 32, "TYRE", 1.0 - leader.tire_wear, GOOD)

            load = leader.corner_load_ratio
            text(
                small,
                f"lap {leader.completed_laps}   cp {leader.next_checkpoint}/"
                f"{len(self.track.meta['checkpoints'])}   "
                f"{leader.g_force:.1f}g   grip {load * 100:3.0f}%",
                16,
                276,
                WARN if load > 0.85 else DIM,
            )

        surface.blit(panel, (24, 24))

    def _bar(self, panel, x, y, width, label, value, colour):
        import pygame

        small = self.fonts[0]
        value = max(0.0, min(1.0, value))
        panel.blit(small.render(label, True, DIM), (x, y - 2))
        track_rect = pygame.Rect(x + 78, y + 1, width - 78 - 44, 12)
        pygame.draw.rect(panel, (46, 50, 60), track_rect, border_radius=3)
        if value > 0:
            filled = track_rect.copy()
            filled.width = max(2, int(track_rect.width * value))
            shade = colour if value > 0.2 else WARN
            pygame.draw.rect(panel, shade, filled, border_radius=3)
        panel.blit(
            small.render(f"{value * 100:3.0f}%", True, INK),
            (x + width - 38, y - 2),
        )

    @staticmethod
    def _fmt(value):
        return "-" if value == float("-inf") else f"{value:.1f}"


# --------------------------------------------------------------------------


def replay(args, cfg, track, screen, world, fonts):
    """Drive a saved genome around the track on repeat."""
    import neat
    import pygame

    neat_config = load_neat_config(args.config)
    with open(args.replay, "rb") as handle:
        genome = pickle.load(handle)
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    sim = Simulation(args, cfg, track, screen, world, fonts)
    sim.round_label = "RUN"
    clock = pygame.time.Clock()
    meta = track.car_meta()

    while True:
        sim.generation += 1
        x, y, angle = track.grid_slots(1, cfg)[0]
        car = Car(x, y, angle, meta, cfg)
        fitness = 0.0
        for tick in range(args.ticks):
            if sim.handle_events():
                break
            car.update(track.surface, track.border_mask, net.activate(car.get_data()),
                       None, tick)
            if not car.is_alive():
                break
            fitness += car.get_reward()
            sim.best_laps = max(sim.best_laps, car.completed_laps)
            sim.best_fitness = max(sim.best_fitness, fitness)
            if tick % sim.ticks_per_frame == 0:
                sim.render([car], tick, 1, 1)
                clock.tick(args.fps)
        print(
            f"  run {sim.generation:>3} | {fitness:9.1f} | "
            f"{car.completed_laps} laps | "
            f"{'finished the run' if car.is_alive() else 'crashed'} "
            f"after {car.time} ticks"
        )


# --------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame

    if args.seed is not None:
        import random

        random.seed(args.seed)

    pygame.init()
    # Deliberately not make_cfg(fps=args.fps): sim_config's `fps` is a physics
    # constant, the one that turns px/tick into m/s for the force model. Letting
    # a render cap move it would rescale every speed, g-force and grip limit.
    cfg = make_cfg()

    try:
        track = Track(args.track)
    except FileNotFoundError as error:
        raise SystemExit(str(error))

    world_size = track.resolution
    cfg["screen_width"], cfg["screen_height"] = world_size

    if args.headless:
        screen = pygame.display.set_mode((1, 1))
        world = pygame.Surface(world_size)
    else:
        scale = args.scale
        if scale is None:
            try:
                desktop = pygame.display.get_desktop_sizes()[0]
                scale = min(
                    1.0, 0.92 * desktop[0] / world_size[0], 0.92 * desktop[1] / world_size[1]
                )
            except Exception:
                scale = 1.0
        window = (int(world_size[0] * scale), int(world_size[1] * scale))
        screen = pygame.display.set_mode(window)
        pygame.display.set_caption(f"Formula NEAT - {track.name}")
        world = screen if window == world_size else pygame.Surface(world_size)

    track.load(cfg["border_color"])
    cfg["car_sprite_path"] = os.path.join(ROOT, cfg["car_sprite_path"])

    mono = "menlo,consolas,dejavusansmono,couriernew,monospace"
    fonts = (
        pygame.font.SysFont(mono, 14),
        pygame.font.SysFont(mono, 16),
        pygame.font.SysFont(mono, 28, bold=True),
    )

    if args.replay:
        return replay(args, cfg, track, screen, world, fonts)

    if args.save_best is None:
        args.save_best = os.path.join(ROOT, f"best_{track.name}.pkl")

    import neat

    neat_config = load_neat_config(args.config)
    population = neat.Population(neat_config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    print(
        f"\nFormula NEAT | track {track.name} ({track.length_px}px lap, "
        f"{len(track.meta['checkpoints'])} checkpoints) | "
        f"pop {neat_config.pop_size} | {args.ticks} ticks/gen"
        f"{' | headless' if args.headless else ''}\n"
    )

    sim = Simulation(args, cfg, track, screen, world, fonts)
    population.run(sim.evaluate, args.generations)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)

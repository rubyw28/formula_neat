# Formula NEAT baseline adapted from NeuralNine/ai-car-simulation.
# Current stage: baseline driving loop + 2026 Step 1 Active Aero (X/Z).

import math
import os
import sys

import neat
import pygame

# Constants
WIDTH = 1920
HEIGHT = 1080

# Scale car up (~4x from current values).
CAR_SIZE_X = 40
CAR_SIZE_Y = 20

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

current_generation = 0  # Generation counter
MIN_DEAD_TICKS_BEFORE_ADVANCE = 20  # Small grace window before skipping dead generation
# 2026 Step 1: in X mode, turning at/above this speed causes a crash.
AERO_X_CRASH_SPEED = 16
HUD_MODE_HOLD_TICKS = 15  # Hold displayed aero mode to reduce text flicker
displayed_aero_mode = "Z"
displayed_aero_mode_since = 0

# 2026 Step 2: simplified battery deploy/harvest model
BATTERY_MAX = 100.0
BATTERY_DEPLOY_PER_TICK = 1.0
BATTERY_HARVEST_BRAKE = 1.4
BATTERY_HARVEST_COAST = 0.25
BATTERY_MIN_FOR_BOOST = 1.0

# 2026 Step 3: manual override (available when close to another car)
OVERRIDE_DISTANCE_PX = 90
OVERRIDE_BONUS_SPEED = 2
OVERRIDE_DRAIN_PER_TICK = 2.2


class Car:
    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load("assets/cars/car.png").convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [830, 920]  # Original baseline starting position
        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed
        # 2026 Step 1 state
        self.aero_mode = "Z"  # Z = higher downforce cornering mode
        self.turn_input = 0  # -1 left, 0 straight, +1 right
        # 2026 Step 2 state
        self.battery_energy = BATTERY_MAX
        self.energy_flow = "IDLE"  # DEPLOY / HARVEST / IDLE
        # 2026 Step 3 state
        self.override_active = False
        self.override_eligible = False

    def draw(self, screen):
        # Draw rotated sprite centered on the car center.
        draw_rect = self.rotated_sprite.get_rect()
        draw_rect.center = (int(self.center[0]), int(self.center[1]))
        screen.blit(self.rotated_sprite, draw_rect.topleft)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + degree))) * length
        )
        y = int(
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + degree))) * length
        )

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max)
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * length
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * length
            )

        # Calculate Distance To Border And Append To Radars List
        dist = int(
            math.sqrt(
                math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)
            )
        )
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Calculate Four Corners, length is half the side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check track-border collisions.
        self.check_collision(game_map)
        # 2026 Active Aero rule (first incremental slice):
        # X mode is straightline-oriented and unstable for fast turning.
        if (
            self.alive
            and self.aero_mode == "X"
            and self.turn_input != 0
            and self.speed >= AERO_X_CRASH_SPEED
        ):
            self.alive = False
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            return_values[i] = int(radar[1] / 30)
        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Safe rotation path for non-square sprites.
        return pygame.transform.rotate(image, angle)


def run_simulation(genomes, config):
    nets = []
    cars = []

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load("map.png").convert()

    global current_generation, displayed_aero_mode, displayed_aero_mode_since
    current_generation += 1
    displayed_aero_mode = "Z"
    displayed_aero_mode_since = 0

    counter = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)

        # For each car, decode NN action.
        # Action mapping (current incremental 2026 version):
        # 0 left, 1 right, 2 brake, 3 accelerate, 4 toggle X aero mode, 5 override
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            # Primary discrete action comes from first 5 outputs.
            choice = output[:5].index(max(output[:5]))
            override_request = len(output) > 5 and output[5] > 0.3
            # Nearest-car check approximates the "1-second gap" style trigger.
            nearest_distance = None
            for j, other in enumerate(cars):
                if i == j or not other.is_alive():
                    continue
                dx = car.center[0] - other.center[0]
                dy = car.center[1] - other.center[1]
                d = math.sqrt(dx * dx + dy * dy)
                if nearest_distance is None or d < nearest_distance:
                    nearest_distance = d
            car.override_eligible = (
                nearest_distance is not None and nearest_distance <= OVERRIDE_DISTANCE_PX
            )
            car.override_active = False
            car.turn_input = 0
            car.aero_mode = "Z"
            if choice == 0:
                car.angle += 10  # Left
                car.turn_input = -1
            elif choice == 1:
                car.angle -= 10  # Right
                car.turn_input = 1
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow Down
                car.battery_energy = min(
                    BATTERY_MAX, car.battery_energy + BATTERY_HARVEST_BRAKE
                )
                car.energy_flow = "HARVEST"
            elif choice == 3:
                # 50/50 style simplification: 1 speed unit from base power,
                # optional 1 speed unit from battery if enough energy.
                car.speed += 1
                if car.battery_energy >= BATTERY_MIN_FOR_BOOST:
                    car.speed += 1
                    car.battery_energy -= BATTERY_DEPLOY_PER_TICK
                    car.energy_flow = "DEPLOY"
                else:
                    car.energy_flow = "IDLE"
                if (
                    override_request
                    and car.override_eligible
                    and car.battery_energy >= OVERRIDE_DRAIN_PER_TICK
                ):
                    car.speed += OVERRIDE_BONUS_SPEED
                    car.battery_energy -= OVERRIDE_DRAIN_PER_TICK
                    car.override_active = True
                    car.energy_flow = "DEPLOY"
            else:
                car.aero_mode = "X"
                car.battery_energy = min(
                    BATTERY_MAX, car.battery_energy + BATTERY_HARVEST_COAST
                )
                car.energy_flow = "HARVEST"

            if choice in (0, 1):
                car.battery_energy = min(
                    BATTERY_MAX, car.battery_energy + BATTERY_HARVEST_COAST
                )
                car.energy_flow = "HARVEST"

            car.battery_energy = max(0.0, min(BATTERY_MAX, car.battery_energy))

        # Check If Car Is Still Alive
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0 and counter >= MIN_DEAD_TICKS_BEFORE_ADVANCE:
            break

        counter += 1
        if counter == 30 * 15:  # Faster iteration than original (about 7.5s)
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render(
            "Generation: " + str(current_generation), True, (0, 0, 0)
        )
        hud_center_y = HEIGHT // 2
        text_rect = text.get_rect(center=(WIDTH // 2, hud_center_y - 70))
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect(center=(WIDTH // 2, hud_center_y - 42))
        screen.blit(text, text_rect)
        lead_car = None
        lead_distance = -1
        for car in cars:
            if car.is_alive():
                if car.distance > lead_distance:
                    lead_distance = car.distance
                    lead_car = car

        lead_mode = lead_car.aero_mode if lead_car else "Z"
        # Stabilize HUD mode text so it doesn't flash rapidly.
        if lead_mode != displayed_aero_mode and (counter - displayed_aero_mode_since) >= HUD_MODE_HOLD_TICKS:
            displayed_aero_mode = lead_mode
            displayed_aero_mode_since = counter
        mode_text = alive_font.render("Lead Mode: " + displayed_aero_mode, True, (0, 0, 0))
        text_rect = mode_text.get_rect(center=(WIDTH // 2, hud_center_y - 14))
        screen.blit(mode_text, text_rect)

        lead_battery = lead_car.battery_energy if lead_car else 0.0
        lead_flow = lead_car.energy_flow if lead_car else "IDLE"
        battery_text = alive_font.render(
            f"Battery: {int(lead_battery)}% ({lead_flow})", True, (0, 0, 0)
        )
        text_rect = battery_text.get_rect(center=(WIDTH // 2, hud_center_y + 14))
        screen.blit(battery_text, text_rect)
        lead_dist_text = alive_font.render(
            f"Lead Distance: {int(lead_distance) if lead_car else 0}", True, (0, 0, 0)
        )
        text_rect = lead_dist_text.get_rect(center=(WIDTH // 2, hud_center_y + 42))
        screen.blit(lead_dist_text, text_rect)
        override_text = alive_font.render(
            "Override: "
            + (
                "ON"
                if lead_car and lead_car.override_active
                else ("READY" if lead_car and lead_car.override_eligible else "OFF")
            ),
            True,
            (0, 0, 0),
        )
        text_rect = override_text.get_rect(center=(WIDTH // 2, hud_center_y + 70))
        screen.blit(override_text, text_rect)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    # Frozen baseline: config.txt now belongs to the car.py physics model.
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config_baseline.txt"
    )
    neat_config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    population = neat.Population(neat_config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.run(run_simulation, 1000)

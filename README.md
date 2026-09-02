# Formula NEAT

NEAT self-driving cars racing under simplified 2026 F1 mechanics: active aero,
battery deployment, manual override, tyre wear, and a grip model that crashes a
car when it asks more of the tyres than they have.

## Quick start

```powershell
pip install pygame neat-python

python main.py                    # train on the oval, with a window
python main.py --track monza      # train on Monza
python main.py --headless -g 200  # train fast, no rendering
python main.py --replay best_oval.pkl
```

The best genome of each run is written to `best_<track>.pkl` whenever it
improves, so `--replay` picks up where training left off.

While a window is open:

| key     | effect                                    |
| ------- | ----------------------------------------- |
| `Esc`   | quit                                      |
| `Space` | end the current generation                |
| `D`     | debug overlay: radars, racing line, checkpoints |
| `H`     | hide the telemetry panel                  |
| `1`-`5` | run 1-5 simulation ticks per drawn frame  |

## Layout

| file                    | role |
| ----------------------- | ---- |
| `main.py`               | training loop, rendering, telemetry HUD, replay |
| `car.py`                | the car: physics, sensors, power unit, fitness |
| `sim_config.py`         | every physics, power-unit, sensor and fitness constant |
| `track.py`              | track image, collision mask, starting grid |
| `config.txt`            | NEAT settings for the current model (13 in, 5 out) |
| `tools/build_tracks.py` | derives track metadata from a track image |
| `tools/preview_track.py`| draws that metadata over the image, to check it |
| `tools/calibrate.py`    | what the physics constants mean on a real track |
| `newcar.py`             | the original baseline, frozen (see below) |
| `config_baseline.txt`   | NEAT settings for that baseline (5 in, 6 out) |

## Tracks

A track is an image plus a JSON file. The image is white where the car crashes
and non-white where it can drive; the drivable region must be a closed ring that
does not touch the image edge.

The JSON is **generated, not hand-written**:

```powershell
python tools/build_tracks.py
python tools/preview_track.py oval
```

`build_tracks.py` cuts the ring open, releases a breadth-first wavefront around
it, and takes the centroid of each wavefront as a centerline point - which also
puts them in lap order. From that it derives the racing line, evenly spaced
checkpoints sized to the local track width, corner apexes, and a spawn point on
the start/finish line (or, failing that, the middle of the longest straight).

Deriving it beats placing points by hand: the checkpoints that shipped with the
original `monza.json` had 9 of 12 sitting off the track, and the spawn point was
in the middle of the infield.

`monza_custom_topdown.png` is a thin outline of the circuit, so it is not
drivable as drawn. The builder traces it and paints a track band along it,
sizing the band from the circuit's own self-clearance so two straights never
weld together, then derives the metadata from the painted result.

To add a track, drop an image in `assets/tracks/`, add an entry to `TRACKS` in
`tools/build_tracks.py`, and rebuild.

| track   | lap     | width | corners |
| ------- | ------- | ----- | ------- |
| `oval`  | 4029 px | 129 px | 213 px tightest |
| `monza` | 2671 px | 90 px  | 83 px hairpin   |

## 2026 rules modeled

Gameplay approximations, not an FIA-accurate physics model.

**Active aerodynamics (X / Z).** Z is the default, high-downforce mode. X trades
downforce for straight-line speed: a higher top speed, but ~28% less grip and
duller steering. Taking a corner in X at speed ends the lap.

**Manual override, not DRS.** Overtake assistance is an electrical boost that
only unlocks when another car is within 120 px, and it drains the battery about
three times as fast as normal deployment.

**Power unit split.** Roughly half the acceleration comes from the engine and
half from electrical deployment, so a flat battery costs real lap time.
Braking and lifting harvest it back.

**Tyres.** Grip decays with distance and speed, down to about 45% of fresh over
a long stint, which lowers the speed the car can carry through a corner.

### Cornering

`car.py` estimates a turn radius from the steering angle with a bicycle model,
computes the centrifugal force on a 768 kg car at that radius, and crashes it if
that exceeds the grip available from the tyres and the current aero mode. In
practice this is a speed-versus-steering envelope: the faster you go, the less
you may turn.

`tools/calibrate.py` prints that envelope against real track geometry, which is
the sane way to tune `sim_config.py`:

```
=== oval ===
  top speed  Z 9.0px/tick (175 km/h)   X 10.8px/tick (210 km/h)
  tightest corner (r=213px) on fresh Z: 5.40px/tick (105 km/h) - limited by grip
  tightest corner (r=213px) on fresh X: 4.85px/tick ( 94 km/h) - limited by grip
  tightest corner (r=213px) on worn  Z: 4.15px/tick ( 81 km/h) - limited by grip
  clean lap ~543 ticks (9.1s at 60fps)
```

## Network

13 inputs:

| index  | input |
| ------ | ----- |
| 0-4    | radar distances from the nose at -90, -45, 0, +45, +90 degrees |
| 5      | speed, as a fraction of maximum |
| 6      | tyre grip |
| 7      | battery charge |
| 8      | aero mode (1 = X) |
| 9      | override available |
| 10     | cornering load, as a fraction of the grip limit |
| 11-12  | distance and heading error to the next checkpoint |

5 outputs, all continuous, `tanh`-squashed:

| index | output |
| ----- | ------ |
| 0     | steering, -1 to +1 |
| 1     | throttle |
| 2     | brake |
| 3     | aero mode: positive selects X |
| 4     | override request |

This replaces the baseline's pick-one-of-six discrete actions, so a car can
brake and steer at the same time.

## Fitness

Scored per tick in `Car.update`, with every weight in `sim_config.py`:

| term | weight | notes |
| ---- | ------ | ----- |
| checkpoint reached | **+18** | in order; 20 of them on the oval |
| lap completed | **+220** | |
| corner apex hit | **+2.2** | within 25 px of an apex point |
| on the racing line | **+0.30** | fading to 0 at 55 px, negative beyond |
| pace | **+0.10 x speed** | up to +1.08 flat out |
| battery above 35% | **+0.05** | |
| battery below 12% | **-0.35** | |
| every tick | **-0.30** | |
| steering | **-0.01 x \|steer\|** | |

`main.py` also retires a car that has not reached a checkpoint in
`--stall-ticks` (260 by default), so nothing farms reward by idling.

### Why these weights

The original weights paid a car **2.08 per tick just for existing on the
racing line** - 2495 over a generation - while a whole lap of the oval paid
580. Crawling strictly dominated racing, and training converged on exactly
that: the best car after 50 generations drove at 29 km/h and never completed a
lap.

The per-tick terms are now small enough that a car has to cover ground to
score. Over a 1200-tick generation:

| | per tick | generation total |
| --- | --- | --- |
| crawling on the line | 0.20 | ~380 |
| racing at 6.5 px/tick | 0.70 | ~1740 |

## Baseline

`newcar.py` is the original discrete-action version, kept runnable for
comparison. It has its own inline car class, its own 5-input NEAT config, and
its own simpler rules (X mode crashes on *any* turn above a speed threshold,
battery as a 0-100 counter). It is frozen - new work goes in `car.py` and
`main.py`.

```powershell
python .\newcar.py
```

## Notes on the physics model

Four things in `car.py` needed fixing or are worth knowing about.

**Sprite rotation crashed the simulation.** `rotate_center` cropped the rotated
sprite back to the unrotated rectangle, which is impossible past about 45
degrees - a rotated 34x16 car is only 16 px wide at 90 degrees - so it raised
`ValueError` on the first real steering input. It now keeps the whole rotated
image, and `sprite_rect()` re-centres it on the car; the collision mask uses the
same rectangle.

**Override cooldown self-cancels.** `car.py` stamps `last_override_tick` on
every tick the override is *active*, so any non-zero `override_cooldown_ticks`
makes the override fire for one tick and then lock itself out. It is set to `0`
in `sim_config.py` and battery drain is the real limiter. A proper fix would
stamp only when the override *starts*.

**Radars stride.** Walking one pixel at a time while recomputing `sin` and `cos`
per pixel - up to 260 steps, 5 radars, per car, per tick - was by far the most
expensive thing in the simulation. The trig is now hoisted out of the loop, and
the ray strides `radar_coarse_step` (4) pixels at a time before walking the last
stride exactly. Measured over 2000 readings, 0.9% differ from the pixel-by-pixel
result, worst case by 6 px out of 260. Set `radar_coarse_step` to 1 to disable.

**The nearest racing-line point is found locally.** A car moves a few pixels per
tick, so the search starts from last tick's index and scans a window around it,
falling back to a full sweep if the best sits on the window edge.

Together those two changes are worth about **3x**: 4800 to 14700 car-ticks per
second on this machine.

## Requirements

- Python 3.9+
- `pygame`, `neat-python`
- `numpy`, but only for `tools/build_tracks.py`. The simulation does not need it.

## Credits

Inspired by [NeuralNine/ai-car-simulation](https://github.com/NeuralNine/ai-car-simulation).

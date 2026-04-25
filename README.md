# Formula NEAT

NEAT self-driving car simulation with incremental 2026 F1 mechanics.

## Current State

The project is intentionally kept close to baseline, with three added features:
- **Active Aero (Step 1)**  
  `X` / `Z` mode toggle, where `X` is unsafe for high-speed turns.
- **Battery Energy (Step 2)**  
  simplified deploy/harvest model with HUD telemetry.
- **Manual Override (Step 3)**  
  extra electric boost when a car is close to another car.

## 2026 F1 Rules Modeled

This simulation currently models simplified versions of key 2026-era concepts:
- **Active Aerodynamics (`X`/`Z`)**  
  `X` mode is treated as low-drag / low-downforce, while `Z` is the default safer mode.
- **No DRS workflow**  
  Overtake assistance is represented through a manual electrical override path instead of classic DRS behavior.
- **Power Unit Energy Usage**  
  Acceleration can consume battery energy, while braking/coasting harvests energy back.
- **Manual Override Trigger**  
  Override is only available when a car is sufficiently close to another car and has enough battery.

Notes:
- This is a gameplay-oriented approximation, not a full FIA-accurate physics model.
- The next planned steps are deeper grip/force modeling and full modularization into `car.py`.

## Files

- `newcar.py` - active simulation loop and car logic
- `config.txt` - NEAT network/evolution settings
- `map.png` - active track
- `assets/cars/car.png` - active car sprite
- `car.py` - reserved for upcoming modular refactor

## Requirements

- Python 3.10+
- `pygame`
- `neat-python`

Install:

```powershell
pip install pygame neat-python
```

## Run

```powershell
python .\newcar.py
```

Controls:
- `Esc` to quit
- Window close button also exits

## Action Mapping (Current)

The NEAT output layer currently uses 6 actions:
- `0` -> turn left
- `1` -> turn right
- `2` -> brake / harvest energy
- `3` -> accelerate (base + battery assist when available)
- `4` -> set `X` aero mode
- `5` -> request manual override (only applies when close enough + enough battery)

## HUD (Current)

- Generation
- Still Alive
- Lead Mode (`X` / `Z`)
- Battery percent + flow (`DEPLOY`, `HARVEST`, `IDLE`)
- Lead distance
- Override status (`ON`, `READY`, `OFF`)

## Fast Tuning Tips

If behavior is unstable, tweak in `newcar.py`:
- Spawn position (`self.position`)
- Car size (`CAR_SIZE_X`, `CAR_SIZE_Y`)
- X-mode crash threshold (`AERO_X_CRASH_SPEED`)
- Battery constants (`BATTERY_*`)
- Override constants (`OVERRIDE_*`)
- Generation pacing (`counter == 30 * 15`)

## Credits

This project is inspired by the original:
- [NeuralNine/ai-car-simulation](https://github.com/NeuralNine/ai-car-simulation)

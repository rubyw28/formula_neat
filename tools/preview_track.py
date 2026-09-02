"""Render a track's generated metadata over its image so it can be eyeballed.

    python tools/preview_track.py oval --out preview_oval.png

Draws the racing line, checkpoint circles in lap order, apex markers and the
spawn arrow. Use it after `build_tracks.py` to confirm the derived geometry
actually sits on the track.
"""

import argparse
import json
import math
import os
import sys

import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(meta, scale):
    surface = pygame.image.load(os.path.join(ROOT, meta["image"])).convert()
    line = meta["racing_line"]
    apexes = set(meta["apex_indices"])

    for i in range(len(line)):
        pygame.draw.line(
            surface, (255, 60, 60), line[i], line[(i + 1) % len(line)], 3
        )
    for i, point in enumerate(line):
        if i in apexes:
            pygame.draw.circle(surface, (255, 210, 0), point, 11)
            pygame.draw.circle(surface, (0, 0, 0), point, 11, 2)

    font = pygame.font.Font(None, 30)
    for i, cp in enumerate(meta["checkpoints"]):
        pygame.draw.circle(surface, (0, 200, 255), (cp["x"], cp["y"]), cp["radius"], 3)
        label = font.render(str(i), True, (0, 200, 255))
        surface.blit(label, (cp["x"] + 6, cp["y"] + 6))

    spawn = meta["spawn"]
    heading = math.radians(360 - spawn["angle"])
    tip = (
        spawn["x"] + math.cos(heading) * 90,
        spawn["y"] + math.sin(heading) * 90,
    )
    pygame.draw.line(surface, (0, 255, 0), (spawn["x"], spawn["y"]), tip, 6)
    pygame.draw.circle(surface, (0, 255, 0), (spawn["x"], spawn["y"]), 10)

    if scale != 1.0:
        size = (int(surface.get_width() * scale), int(surface.get_height() * scale))
        surface = pygame.transform.smoothscale(surface, size)
    return surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track")
    parser.add_argument("--out", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_mode((1, 1), pygame.HIDDEN)

    path = os.path.join(ROOT, "assets", "tracks", f"{args.track}.json")
    with open(path) as handle:
        meta = json.load(handle)

    out = args.out or os.path.join(ROOT, f"preview_{args.track}.png")
    pygame.image.save(render(meta, args.scale), out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

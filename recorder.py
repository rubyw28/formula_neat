"""Record the simulation to an animated GIF.

Used by `main.py --record demo.gif`. Needs Pillow, which the simulation itself
does not - if it is missing, recording is refused with a clear message rather
than failing halfway through a run.

Frames are quantised to a palette taken from the first frame and reused for the
rest. The palette barely changes between frames - a black track, a pale
background, cyan cars and a dark HUD - so reusing it keeps colours stable, keeps
memory flat, and produces a much smaller file than quantising each frame
independently.
"""

import os

import pygame

MAX_COLOURS = 64


class Recorder:
    """Collects frames from a pygame surface and writes them as a GIF."""

    def __init__(self, path, every=3, max_frames=260, width=760, fps=60):
        try:
            from PIL import Image
        except ImportError:
            raise SystemExit(
                "--record needs Pillow, which the simulation itself does not.\n"
                "    pip install Pillow"
            )
        self._Image = Image
        self.path = path
        self.every = max(1, every)
        self.max_frames = max_frames
        self.width = width
        # Output frame rate, and so how long each GIF frame is held.
        self.duration_ms = max(20, int(round(1000.0 * self.every / fps)))

        self.frames = []
        self.palette = None
        self._ticks = 0
        self.full = False

    def capture(self, surface):
        """Take a frame, if this tick is due. Returns True while still recording."""
        if self.full:
            return False
        self._ticks += 1
        if (self._ticks - 1) % self.every:
            return True

        height = int(round(surface.get_height() * self.width / surface.get_width()))
        small = pygame.transform.smoothscale(surface, (self.width, height))
        image = self._Image.frombytes(
            "RGB", small.get_size(), pygame.image.tostring(small, "RGB")
        )

        if self.palette is None:
            self.palette = image.convert(
                "P", palette=self._Image.Palette.ADAPTIVE, colors=MAX_COLOURS
            )
            self.frames.append(self.palette)
        else:
            self.frames.append(image.quantize(palette=self.palette, dither=0))

        if len(self.frames) >= self.max_frames:
            self.full = True
            return False
        return True

    def save(self):
        """Write the GIF. Returns its path, or None if nothing was captured."""
        if not self.frames:
            return None
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.frames[0].save(
            self.path,
            save_all=True,
            append_images=self.frames[1:],
            duration=self.duration_ms,
            loop=0,
            optimize=True,
            disposal=1,
        )
        return self.path

    def summary(self):
        size = os.path.getsize(self.path) / 1e6 if os.path.exists(self.path) else 0.0
        seconds = len(self.frames) * self.duration_ms / 1000.0
        return (
            f"{self.path}: {len(self.frames)} frames, {seconds:.1f}s, "
            f"{size:.1f} MB"
        )

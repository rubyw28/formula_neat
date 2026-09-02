"""Derive track metadata (centerline, checkpoints, apexes, spawn) from a track image.

Convention shared with the simulation: a track image is white where the car
crashes and non-white where it can drive. The drivable region must be a closed
ring that does not touch the image edge.

This is a one-off authoring tool, not part of the simulation runtime. It is the
only thing in the project that needs numpy.

    python tools/build_tracks.py            # rebuild every track
    python tools/build_tracks.py --track oval

The Monza source art is a thin outline of the circuit, so it is not drivable as
drawn. For that track we first trace the outline into an ordered centerline and
paint a real track band along it, then derive metadata from the painted result
using the same code path as any other track.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHITE_CUTOFF = 200  # >= this on every channel counts as off-track white


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------

def load_rgb(path):
    surface = pygame.image.load(path)
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def drivable_mask(rgb):
    """True where the car may drive (anything that is not near-white)."""
    return ~(rgb >= WHITE_CUTOFF).all(axis=2)


def bfs_distance(passable, seed_indices):
    """4-connected BFS distance over `passable`, in pixels. -1 where unreachable.

    Vectorised by wavefront: each iteration expands the whole frontier at once,
    so cost scales with track length rather than pixel count.
    """
    height, width = passable.shape
    flat = passable.ravel()
    dist = np.full(height * width, -1, np.int32)

    frontier = np.asarray(seed_indices, np.int64)
    frontier = frontier[flat[frontier]]
    dist[frontier] = 0

    step = 0
    while frontier.size:
        step += 1
        neighbours = np.concatenate(
            (frontier - 1, frontier + 1, frontier - width, frontier + width)
        )
        neighbours = neighbours[(neighbours >= 0) & (neighbours < height * width)]
        neighbours = neighbours[flat[neighbours] & (dist[neighbours] < 0)]
        if neighbours.size == 0:
            break
        neighbours = np.unique(neighbours)
        dist[neighbours] = step
        frontier = neighbours
    return dist.reshape(height, width)


def edge_distance(band):
    """Distance from each drivable pixel to the nearest off-track pixel."""
    rim = np.zeros_like(band)
    rim[1:] |= ~band[:-1]
    rim[:-1] |= ~band[1:]
    rim[:, 1:] |= ~band[:, :-1]
    rim[:, :-1] |= ~band[:, 1:]
    seeds = np.flatnonzero((rim & band).ravel())
    return bfs_distance(band, seeds)


def flood_from_border(mask):
    """Component of `mask` reachable from the image edge."""
    height, width = mask.shape
    edge = np.zeros_like(mask)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    seeds = np.flatnonzero((edge & mask).ravel())
    return bfs_distance(mask, seeds) >= 0


# --------------------------------------------------------------------------
# centerline extraction
# --------------------------------------------------------------------------

def _nearest_off_track_direction(band, point, search_radius):
    """Unit vector from `point` towards the closest off-track pixel."""
    y0, x0 = point
    height, width = band.shape
    best = None
    best_d2 = None
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            y, x = y0 + dy, x0 + dx
            if not (0 <= y < height and 0 <= x < width) or band[y, x]:
                continue
            d2 = dy * dy + dx * dx
            if best_d2 is None or d2 < best_d2:
                best_d2, best = d2, (dy, dx)
    if best is None:
        raise RuntimeError("no off-track pixel near the cut seed")
    norm = math.hypot(best[0], best[1])
    return best[0] / norm, best[1] / norm


def _cut_mask(band, seed, normal, reach):
    """Paint a short wall straight across the track so the ring becomes an arc."""
    height, width = band.shape
    cut = np.zeros_like(band)
    y0, x0 = seed
    for sign in (1, -1):
        for step in range(int(reach * 2) + 1):
            t = sign * step * 0.5
            y = int(round(y0 + normal[0] * t))
            x = int(round(x0 + normal[1] * t))
            if not (0 <= y < height and 0 <= x < width):
                break
            cut[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
            # Keep going a little past the far edge so the wall really severs.
            if step > reach and not band[y, x]:
                break
    return cut


def ordered_centerline(band, seed=None, normal=None):
    """Return centerline points, in travel order, one per unit of track length.

    The ring is cut open with a wall across the track, then a breadth-first
    wavefront is released from one face of the wall. Every wavefront level is a
    cross-section of the track, so its centroid is a centerline point and the
    level index orders them around the lap.
    """
    inset = edge_distance(band)
    if seed is None:
        seed_flat = int(np.argmax(inset))
        seed = (seed_flat // band.shape[1], seed_flat % band.shape[1])
    reach = int(inset[seed]) + 2

    if normal is None:
        normal = _nearest_off_track_direction(band, seed, reach + 2)
    cut = _cut_mask(band, seed, normal, reach * 3)
    arc = band & ~cut

    # Seed the wavefront from the wall face on one side only.
    tangent = (-normal[1], normal[0])
    touching = np.zeros_like(arc)
    touching[1:] |= cut[:-1]
    touching[:-1] |= cut[1:]
    touching[:, 1:] |= cut[:, :-1]
    touching[:, :-1] |= cut[:, 1:]
    ys, xs = np.nonzero(touching & arc)
    side = (ys - seed[0]) * tangent[0] + (xs - seed[1]) * tangent[1]
    keep = side > 0
    if keep.sum() < 3:
        keep = side < 0
    seeds = (ys[keep] * band.shape[1] + xs[keep]).astype(np.int64)

    dist = bfs_distance(arc, seeds)
    reached = dist >= 0
    if not reached.any():
        raise RuntimeError("track cut failed: nothing reachable")

    ys, xs = np.nonzero(reached)
    levels = dist[ys, xs]
    count = np.bincount(levels)
    sum_y = np.bincount(levels, weights=ys)
    sum_x = np.bincount(levels, weights=xs)
    valid = count > 0
    points = np.stack([sum_x[valid] / count[valid], sum_y[valid] / count[valid]], axis=1)

    coverage = reached.sum() / band.sum()
    if coverage < 0.90:
        raise RuntimeError(
            "track cut severed the ring in more than one place "
            f"(only {coverage:.0%} of the track was reached)"
        )
    return points


def centerline_of(band):
    """Centerline of a ring-shaped track, seam included.

    Cutting the ring leaves a small discontinuity where the wavefront starts and
    ends. A first pass locates the track's longest straight, then a second pass
    re-cuts there, square across the track, so the seam falls somewhere it does
    no harm instead of chording across a corner.
    """
    rough = smooth_closed(ordered_centerline(band), 21)
    rough, _ = resample_closed(rough, 12.0)
    rough = smooth_closed(rough, 9)

    index = longest_straight_center(rough, exclude=max(2, len(rough) // 8))
    tangent = tangents_of(rough)[index]
    point = rough[index]

    height, width = band.shape
    seed = (int(round(point[1])), int(round(point[0])))
    if not (0 <= seed[0] < height and 0 <= seed[1] < width and band[seed]):
        # The chosen point is not on the track, so the rough pass is untrustworthy
        # here. Keep the first-pass result rather than cutting somewhere arbitrary.
        return ordered_centerline(band)

    # Perpendicular to travel, in (row, col) order.
    normal = (float(tangent[0]), float(-tangent[1]))
    return ordered_centerline(band, seed=seed, normal=normal)


def smooth_closed(points, window):
    """Moving average around a closed loop."""
    if window < 3:
        return points
    kernel = np.ones(window) / window
    padded = np.concatenate([points[-window:], points, points[:window]])
    smoothed = np.stack(
        [np.convolve(padded[:, i], kernel, mode="same") for i in range(2)], axis=1
    )
    return smoothed[window:-window]


def resample_closed(points, spacing):
    """Resample a closed polyline to roughly even spacing."""
    loop = np.vstack([points, points[:1]])
    segments = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segments)])
    total = arc[-1]
    count = max(16, int(round(total / spacing)))
    targets = np.linspace(0.0, total, count, endpoint=False)
    out = np.stack(
        [np.interp(targets, arc, loop[:, i]) for i in range(2)], axis=1
    )
    return out, total


def track_width_at(band, point, tangent, limit):
    """Measure the drivable width perpendicular to travel, in pixels."""
    height, width = band.shape
    normal = (-tangent[1], tangent[0])
    reach = []
    for sign in (1, -1):
        step = 0.0
        while step < limit:
            step += 0.5
            x = int(round(point[0] + normal[0] * sign * step))
            y = int(round(point[1] + normal[1] * sign * step))
            if not (0 <= y < height and 0 <= x < width) or not band[y, x]:
                break
        reach.append(step)
    return reach[0] + reach[1]


def tangents_of(points):
    ahead = np.roll(points, -1, axis=0)
    behind = np.roll(points, 1, axis=0)
    vectors = ahead - behind
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def curvature_of(points):
    """Heading change per point, as a positive magnitude."""
    tangents = tangents_of(points)
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    delta = np.diff(np.concatenate([headings, headings[:1]]))
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return np.abs(delta)


def longest_straight_center(points, exclude=0):
    """Index at the middle of the longest run of near-straight track.

    `exclude` blanks that many points either side of index 0. The seam left by
    cutting the ring is dead straight, so without this it would win the search.
    """
    curve = curvature_of(points)
    straight = curve < max(np.percentile(curve, 35), 0.012)
    n = len(points)
    if exclude:
        straight[:exclude] = False
        straight[-exclude:] = False
    best_len, best_end = 0, 0
    run = 0
    # Two passes so a run that wraps past index 0 is still measured whole.
    for i in range(2 * n):
        run = run + 1 if straight[i % n] else 0
        if run > best_len:
            best_len, best_end = run, i
    if best_len == 0:
        return 0
    return (best_end - best_len // 2) % n


def pick_apexes(points, min_gap):
    """Corner apexes: the sharpest point of each sustained curvature peak."""
    curve = curvature_of(points)
    smooth = np.convolve(
        np.concatenate([curve[-5:], curve, curve[:5]]), np.ones(5) / 5, mode="same"
    )[5:-5]
    threshold = max(smooth.mean() * 1.35, np.percentile(smooth, 70))

    apexes = []
    for i in np.argsort(-smooth):
        if smooth[i] < threshold:
            break
        n = len(points)
        if all(min((abs(i - j)), n - abs(i - j)) >= min_gap for j in apexes):
            apexes.append(int(i))
    return sorted(apexes)


# --------------------------------------------------------------------------
# metadata build
# --------------------------------------------------------------------------

def build_meta(image_path, name, line_spacing, checkpoint_spacing, spawn_hint=None):
    rgb = load_rgb(image_path)
    band = drivable_mask(rgb)
    height, width = band.shape

    if band[0].any() or band[-1].any() or band[:, 0].any() or band[:, -1].any():
        raise RuntimeError(f"{image_path}: track touches the image edge")

    raw = smooth_closed(centerline_of(band), 21)
    line, length = resample_closed(raw, line_spacing)
    line = smooth_closed(line, 5)
    tangents = tangents_of(line)

    widths = np.array(
        [track_width_at(band, line[i], tangents[i], 400) for i in range(len(line))]
    )

    # Start on the painted start/finish line if the art has one, otherwise in
    # the middle of the longest straight.
    start = None
    if spawn_hint == "green":
        green = (
            (rgb[:, :, 1].astype(int) - rgb[:, :, 0] > 25)
            & (rgb[:, :, 1].astype(int) - rgb[:, :, 2] > 25)
        )
        if green.any():
            ys, xs = np.nonzero(green)
            marker = np.array([xs.mean(), ys.mean()])
            start = int(np.argmin(np.linalg.norm(line - marker, axis=1)))
    if start is None:
        start = longest_straight_center(line)

    line = np.roll(line, -start, axis=0)
    tangents = np.roll(tangents, -start, axis=0)
    widths = np.roll(widths, -start)

    step = max(1, int(round(checkpoint_spacing / line_spacing)))
    checkpoints = []
    for i in range(0, len(line), step):
        checkpoints.append(
            {
                "x": int(round(line[i][0])),
                "y": int(round(line[i][1])),
                "radius": int(round(min(max(widths[i] * 0.5, 26.0), 75.0))),
            }
        )

    spawn_angle = -math.degrees(math.atan2(tangents[0][1], tangents[0][0])) % 360

    return {
        "name": name,
        "image": os.path.relpath(image_path, ROOT).replace(os.sep, "/"),
        "resolution": [int(width), int(height)],
        "generated_by": "tools/build_tracks.py",
        "spawn": {
            "x": int(round(line[0][0])),
            "y": int(round(line[0][1])),
            "angle": round(spawn_angle, 2),
        },
        "length_px": int(round(length)),
        "median_width_px": int(round(float(np.median(widths)))),
        "checkpoints": checkpoints,
        "racing_line": [[int(round(p[0])), int(round(p[1]))] for p in line],
        # Apexes must be at least ~250px of track apart so one corner
        # does not register as several.
        "apex_indices": pick_apexes(line, min_gap=max(4, int(250 / line_spacing))),
    }


# --------------------------------------------------------------------------
# Monza: turn the outline art into a drivable track
# --------------------------------------------------------------------------

def min_self_clearance(points, ignore_span):
    """Closest approach between two parts of the lap that are far apart along it.

    This is what limits how wide the track can be painted: half of it, less a
    margin, is the widest band that will not weld two straights together.
    """
    coords = np.asarray(points, float)
    n = len(coords)
    index = np.arange(n)
    along = np.abs(index[:, None] - index[None, :])
    along = np.minimum(along, n - along)

    gaps = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    gaps[along <= ignore_span] = np.inf
    return float(gaps.min())


def paint_track_from_outline(outline_path, out_path, half_width):
    """Trace the outline art and paint a drivable track band along it.

    Returns the half-width actually used. The requested width is capped by the
    circuit's own self-clearance, then reduced further if the painted result
    fails to reproduce the source lap.
    """
    rgb = load_rgb(outline_path)
    band = drivable_mask(rgb)
    center = smooth_closed(centerline_of(band), 15)
    center, source_length = resample_closed(center, 6.0)
    center = smooth_closed(center, 9)

    spacing = source_length / len(center)
    clearance = min_self_clearance(center, ignore_span=int(200 / spacing))
    half_width = min(half_width, int(clearance / 2) - 6)

    width, height = rgb.shape[1], rgb.shape[0]
    points = [(int(round(p[0])), int(round(p[1]))) for p in center]

    while half_width >= 20:
        surface = pygame.Surface((width, height))
        surface.fill((255, 255, 255))
        for i in range(len(points)):
            a, b = points[i], points[(i + 1) % len(points)]
            pygame.draw.line(surface, (0, 0, 0), a, b, half_width * 2)
            pygame.draw.circle(surface, (0, 0, 0), a, half_width)

        painted = drivable_mask(
            np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        )
        if _reproduces_lap(painted, source_length):
            pygame.image.save(surface, out_path)
            return half_width
        half_width -= 5
    raise RuntimeError("could not paint a self-avoiding track from the outline")


def _reproduces_lap(band, source_length):
    """The painted track is only good if a lap of it still matches the source.

    A band that welds itself together stays a valid ring, so topology alone is
    not enough - the giveaway is that the surviving lap gets much shorter.
    """
    if not _is_clean_ring(band):
        return False
    try:
        lap = _polyline_length(
            [tuple(p) for p in resample_closed(smooth_closed(centerline_of(band), 21), 12.0)[0]]
        )
    except RuntimeError:
        return False
    return lap > source_length * 0.90


def _polyline_length(points):
    return sum(
        math.hypot(
            points[(i + 1) % len(points)][0] - points[i][0],
            points[(i + 1) % len(points)][1] - points[i][1],
        )
        for i in range(len(points))
    )


def _is_clean_ring(band):
    """A valid track: touches no image edge, and encloses exactly one infield."""
    if band[0].any() or band[-1].any() or band[:, 0].any() or band[:, -1].any():
        return False
    off_track = ~band
    outfield = flood_from_border(off_track)
    infield = off_track & ~outfield
    if infield.sum() < 5000:
        return False
    # Exactly one infield component: seed from the largest and check it covers all.
    ys, xs = np.nonzero(infield)
    seed = np.array([ys[0] * band.shape[1] + xs[0]], np.int64)
    return (bfs_distance(infield, seed) >= 0).sum() == infield.sum()


# --------------------------------------------------------------------------

TRACKS = {
    "oval": {
        "image": "map.png",
        "line_spacing": 26.0,
        "checkpoint_spacing": 210.0,
        "spawn_hint": "green",
    },
    "monza": {
        "image": "assets/tracks/monza_track.png",
        "outline": "assets/tracks/monza_custom_topdown.png",
        "half_width": 55,
        "line_spacing": 26.0,
        "checkpoint_spacing": 210.0,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=sorted(TRACKS), action="append")
    args = parser.parse_args()

    pygame.init()
    names = args.track or sorted(TRACKS)
    for name in names:
        spec = TRACKS[name]
        image = os.path.join(ROOT, spec["image"])

        if "outline" in spec:
            used = paint_track_from_outline(
                os.path.join(ROOT, spec["outline"]), image, spec["half_width"]
            )
            print(f"{name}: painted {spec['image']} at half-width {used}px")

        meta = build_meta(
            image,
            name,
            spec["line_spacing"],
            spec["checkpoint_spacing"],
            spec.get("spawn_hint"),
        )
        out = os.path.join(ROOT, "assets", "tracks", f"{name}.json")
        with open(out, "w") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")
        print(
            f"{name}: {meta['length_px']}px lap, width ~{meta['median_width_px']}px, "
            f"{len(meta['checkpoints'])} checkpoints, "
            f"{len(meta['apex_indices'])} apexes, "
            f"spawn ({meta['spawn']['x']},{meta['spawn']['y']}) @ {meta['spawn']['angle']}deg "
            f"-> {os.path.relpath(out, ROOT)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

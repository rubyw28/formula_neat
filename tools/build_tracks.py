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

# A 2026-regulation car, and how a real track compares to it. A Formula 1
# track is about 12m wide and a car about 2.0m, so the track is roughly six
# cars wide and 2.13 car-lengths wide. Sizing the car from each track's own
# band keeps that proportion right whatever scale the circuit came out at.
CAR_LENGTH_M = 5.63
CAR_WIDTH_M = 2.00
TRACK_WIDTH_IN_CARS = 6.0
MIN_CAR_WIDTH_PX = 7


def car_for_track(track_width_px):
    """Car size in pixels, and the metre scale that makes it a real F1 car."""
    car_width = max(MIN_CAR_WIDTH_PX, int(round(track_width_px / TRACK_WIDTH_IN_CARS)))
    car_length = int(round(car_width * (CAR_LENGTH_M / CAR_WIDTH_M)))
    return {
        "size_x": car_length,
        "size_y": car_width,
        "length_m": CAR_LENGTH_M,
        "width_m": CAR_WIDTH_M,
        # One pixel in metres, anchored so the car is exactly a real car.
        "pixel_to_meter": round(CAR_LENGTH_M / car_length, 5),
    }


def build_meta(image_path, name, line_spacing, checkpoint_spacing, spawn_hint=None,
               extra=None):
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

    # Evenly spaced around the whole loop, so the last gap matches the rest
    # instead of whatever the loop length happened to leave over. Offset by half
    # a gap so checkpoint 0 is not sitting on the spawn point, where the front
    # row would collect it for free on tick zero.
    count = max(4, int(round(len(line) / (checkpoint_spacing / line_spacing))))
    checkpoints = []
    for k in range(count):
        i = int(round((k + 0.5) * len(line) / count)) % len(line)
        # Radius has to reach past the edge of the track, or a car running wide
        # drives straight past the checkpoint without ever registering it.
        radius = min(max(widths[i] * 0.62, 26.0), 120.0)
        checkpoints.append(
            {
                "x": int(round(line[i][0])),
                "y": int(round(line[i][1])),
                "radius": int(round(radius)),
            }
        )

    spawn_angle = -math.degrees(math.atan2(tangents[0][1], tangents[0][0])) % 360

    median_width = float(np.median(widths))
    car = car_for_track(median_width)

    meta = {
        "name": name,
        "image": os.path.relpath(image_path, ROOT).replace(os.sep, "/"),
        "resolution": [int(width), int(height)],
        "generated_by": "tools/build_tracks.py",
        "car": car,
        "spawn": {
            "x": int(round(line[0][0])),
            "y": int(round(line[0][1])),
            "angle": round(spawn_angle, 2),
        },
        "length_px": int(round(length)),
        "median_width_px": int(round(median_width)),
        "width_in_cars": round(median_width / car["size_y"], 2),
        "lap_m": int(round(length * car["pixel_to_meter"])),
        "checkpoints": checkpoints,
        "racing_line": [[int(round(p[0])), int(round(p[1]))] for p in line],
        # Apexes must be at least ~250px of track apart so one corner
        # does not register as several.
        "apex_indices": pick_apexes(line, min_gap=max(4, int(250 / line_spacing))),
    }
    if extra:
        meta.update(extra)
    return meta


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


def paint_band(points, size, half_width):
    """Paint a drivable band of the given half-width along a centreline."""
    surface = pygame.Surface(size)
    surface.fill((255, 255, 255))
    whole = [(int(round(p[0])), int(round(p[1]))) for p in points]
    for i in range(len(whole)):
        pygame.draw.line(
            surface, (0, 0, 0), whole[i], whole[(i + 1) % len(whole)], half_width * 2
        )
        pygame.draw.circle(surface, (0, 0, 0), whole[i], half_width)
    return surface


def paint_track(points, size, source_length, out_path, half_width, floor):
    """Paint a centreline into a drivable track image, as wide as will fit.

    The requested half-width is first capped by the circuit's own self-clearance,
    then reduced until the painted result still reproduces the source lap.
    Returns the half-width used, or None if even `floor` welds the track shut.
    """
    spacing = source_length / len(points)
    clearance = min_self_clearance(points, ignore_span=max(2, int(200 / spacing)))
    half_width = min(half_width, int(clearance / 2) - 5)

    while half_width >= floor:
        surface = paint_band(points, size, half_width)
        painted = drivable_mask(
            np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        )
        if _reproduces_lap(painted, source_length):
            pygame.image.save(surface, out_path)
            return half_width
        half_width -= 3
    return None


def outline_centerline(outline_path):
    """Centreline traced out of thin outline art, plus the frame it lives in."""
    rgb = load_rgb(outline_path)
    band = drivable_mask(rgb)
    center = smooth_closed(centerline_of(band), 15)
    center, length = resample_closed(center, 6.0)
    center = smooth_closed(center, 9)
    return center, length, (rgb.shape[1], rgb.shape[0])


EARTH_RADIUS_M = 6371000.0


def centerline_from_geojson(path, size, margin):
    """Project a real circuit's GeoJSON centreline into the pixel frame.

    Circuit maps have no canonical rotation, and a tall circuit dropped into a
    16:9 frame wastes most of it, so the loop is turned to whichever orientation
    fits largest.
    """
    with open(path) as handle:
        payload = json.load(handle)
    feature = payload["features"][0]
    geometry = feature["geometry"]
    coords = geometry["coordinates"]
    if geometry["type"] == "MultiLineString":
        coords = max(coords, key=len)
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    # Equirectangular about the circuit's own centroid. Over a few kilometres
    # the distortion is far smaller than the pixel we round to.
    lat0 = sum(point[1] for point in coords) / len(coords)
    scale_lon = math.cos(math.radians(lat0))
    metres = np.array(
        [
            [
                EARTH_RADIUS_M * math.radians(point[0]) * scale_lon,
                -EARTH_RADIUS_M * math.radians(point[1]),  # screen y grows downward
            ]
            for point in coords
        ]
    )
    measured_m = _polyline_length([tuple(p) for p in metres])

    width, height = size
    best = None
    for degrees in range(0, 180):
        angle = math.radians(degrees)
        turned = metres @ np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        span_x = max(float(np.ptp(turned[:, 0])), 1e-6)
        span_y = max(float(np.ptp(turned[:, 1])), 1e-6)
        scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
        if best is None or scale > best[0]:
            best = (scale, degrees, turned)

    scale, degrees, turned = best
    pixels = turned * scale
    pixels[:, 0] += width / 2.0 - (pixels[:, 0].min() + pixels[:, 0].max()) / 2.0
    pixels[:, 1] += height / 2.0 - (pixels[:, 1].min() + pixels[:, 1].max()) / 2.0

    pixels, lap_px = resample_closed(pixels, 5.0)
    pixels = smooth_closed(pixels, 7)

    info = {
        "circuit": feature["properties"].get("Name"),
        "official_length_m": feature["properties"].get("length"),
        "measured_length_m": int(round(measured_m)),
        "rotated_deg": degrees,
    }
    return pixels, lap_px, info


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


# Sampled from the start/finish line painted into map.png, so a generated
# track's line matches the hand-drawn one. Both are safely below the white
# cutoff, so painting them changes nothing about where the car may drive.
START_LINE_LIGHT = (163, 193, 160)
START_LINE_DARK = (20, 77, 0)


def paint_start_line(image_path, spawn, half_width, depth=56, square=14):
    """Paint a checkered start/finish line across the track at the spawn point.

    Only pixels that are already drivable are recoloured, and only in colours
    that stay drivable, so the track's shape and collision mask are untouched -
    this is purely so you can see where the lap begins.
    """
    rgb = load_rgb(image_path)
    band = drivable_mask(rgb)
    height, width = band.shape

    # car.py's angle convention: travel is along (cos(360-a), sin(360-a)).
    heading = math.radians(360 - spawn["angle"])
    along = (math.cos(heading), math.sin(heading))
    across = (-along[1], along[0])

    reach = int(depth + 2 * half_width + 8)
    x0 = max(0, spawn["x"] - reach)
    x1 = min(width, spawn["x"] + reach)
    y0 = max(0, spawn["y"] - reach)
    y1 = min(height, spawn["y"] + reach)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = xs - spawn["x"]
    dy = ys - spawn["y"]
    u = dx * along[0] + dy * along[1]       # distance along the track
    v = dx * across[0] + dy * across[1]     # distance across it

    span = half_width + 8
    on_line = (np.abs(u) <= depth / 2.0) & (np.abs(v) <= span) & band[y0:y1, x0:x1]
    light = (
        (np.floor((u + depth / 2.0) / square) + np.floor((v + span) / square)) % 2 == 0
    )

    patch = rgb[y0:y1, x0:x1]
    patch[on_line & light] = START_LINE_LIGHT
    patch[on_line & ~light] = START_LINE_DARK

    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    pygame.image.save(surface, image_path)
    return int(on_line.sum())


def save_clearance_field(image_path, out_path):
    """Write a greyscale map of how far each track pixel is from a wall.

    The simulation ray-marches its radars through this: from any point it can
    safely jump the distance stored there without passing through a wall, which
    is both exact and far cheaper than testing every pixel along the ray.

    The BFS distance is 4-connected, so it overestimates true (Euclidean)
    distance by up to sqrt(2); car.py divides by 1.45 before stepping, which
    keeps every jump conservative.
    """
    band = drivable_mask(load_rgb(image_path))
    field = edge_distance(band)
    field[~band] = 0
    grey = np.clip(field, 0, 255).astype(np.uint8)

    surface = pygame.surfarray.make_surface(
        np.transpose(np.stack([grey] * 3, axis=2), (1, 0, 2))
    )
    pygame.image.save(surface, out_path)
    return int(grey.max())


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

# Every track is one entry here. `image` is what the simulation loads; the
# other keys say where that image comes from.
#   (none)     hand-drawn art, used as-is
#   outline    thin outline art, traced and painted into a drivable band
#   geojson    a real circuit's centreline, projected and painted
TRACKS = {
    "oval": {
        "image": "map.png",
        "line_spacing": 26.0,
        "checkpoint_spacing": 210.0,
        "spawn_hint": "green",
    },
    "monza_art": {
        "image": "assets/tracks/monza_art.png",
        "outline": "assets/tracks/monza_custom_topdown.png",
        "half_width": 55,
        "line_spacing": 26.0,
        "checkpoint_spacing": 210.0,
    },
}

# The F1 calendar, built from real circuit geometry. Kept in season order.
CALENDAR = [
    ("australia", "au-1953", "Albert Park"),
    ("china", "cn-2004", "Shanghai International Circuit"),
    ("japan", "jp-1962", "Suzuka"),
    ("bahrain", "bh-2002", "Bahrain International Circuit"),
    ("saudi_arabia", "sa-2021", "Jeddah Corniche Circuit"),
    ("miami", "us-2022", "Miami International Autodrome"),
    ("canada", "ca-1978", "Circuit Gilles-Villeneuve"),
    ("monaco", "mc-1929", "Circuit de Monaco"),
    ("spain", "es-1991", "Circuit de Barcelona-Catalunya"),
    ("austria", "at-1969", "Red Bull Ring"),
    ("britain", "gb-1948", "Silverstone"),
    ("belgium", "be-1925", "Spa-Francorchamps"),
    ("hungary", "hu-1986", "Hungaroring"),
    ("netherlands", "nl-1948", "Zandvoort"),
    ("monza", "it-1922", "Autodromo Nazionale Monza"),
    ("madrid", "es-2026", "Madring"),
    ("azerbaijan", "az-2016", "Baku City Circuit"),
    ("singapore", "sg-2008", "Marina Bay Street Circuit"),
    ("usa", "us-2012", "Circuit of the Americas"),
    ("mexico", "mx-1962", "Autodromo Hermanos Rodriguez"),
    ("brazil", "br-1977", "Interlagos"),
    ("las_vegas", "us-2023", "Las Vegas Strip Circuit"),
    ("qatar", "qa-2004", "Losail International Circuit"),
    ("abu_dhabi", "ae-2009", "Yas Marina Circuit"),
]

FRAME = (1920, 1080)
FRAME_MARGIN = 46
# Widest band worth painting, and the narrowest that is still worth driving.
# A real circuit is roughly 480 times longer than it is wide, so a 5km lap drawn
# as ~5000px would have a realistic band about 10px across. Every track here is
# painted far wider than that on purpose; the floor is where there is no room
# left to exaggerate, because the circuit genuinely runs alongside itself.
TARGET_HALF_WIDTH = 48
FLOOR_HALF_WIDTH = 15

for _name, _circuit_id, _label in CALENDAR:
    TRACKS[_name] = {
        "image": f"assets/tracks/{_name}.png",
        "geojson": f"assets/tracks/geo/{_circuit_id}.geojson",
        "label": _label,
        "half_width": TARGET_HALF_WIDTH,
        "line_spacing": 26.0,
        "checkpoint_spacing": 210.0,
    }


def build_one(name, spec):
    """Build one track's image (if generated) and its metadata. None if it fails."""
    image = os.path.join(ROOT, spec["image"])
    generated = "outline" in spec or "geojson" in spec
    extra = None
    used = None

    if "outline" in spec:
        points, source_length, size = outline_centerline(
            os.path.join(ROOT, spec["outline"])
        )
        used = paint_track(points, size, source_length, image,
                           spec["half_width"], FLOOR_HALF_WIDTH)
    elif "geojson" in spec:
        geo = os.path.join(ROOT, spec["geojson"])
        if not os.path.exists(geo):
            print(f"{name}: no geometry at {spec['geojson']} - run tools/fetch_circuits.py")
            return None
        points, source_length, info = centerline_from_geojson(geo, FRAME, FRAME_MARGIN)
        info["label"] = spec.get("label")
        extra = info
        used = paint_track(points, FRAME, source_length, image,
                           spec["half_width"], FLOOR_HALF_WIDTH)

    if generated and used is None:
        print(f"{name}: SKIPPED - the circuit runs too close to itself to paint a "
              f"drivable band wider than {FLOOR_HALF_WIDTH * 2}px without welding shut")
        return None

    meta = build_meta(
        image,
        name,
        spec["line_spacing"],
        spec["checkpoint_spacing"],
        spec.get("spawn_hint"),
        extra,
    )
    if generated:
        # The source has no start/finish line, so draw one where the lap
        # actually begins. After build_meta, so marker and spawn cannot drift.
        paint_start_line(image, meta["spawn"], used)

    clearance = os.path.splitext(image)[0] + "_clearance.png"
    save_clearance_field(image, clearance)
    meta["clearance_image"] = os.path.relpath(clearance, ROOT).replace(os.sep, "/")

    out = os.path.join(ROOT, "assets", "tracks", f"{name}.json")
    with open(out, "w") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=sorted(TRACKS), action="append")
    args = parser.parse_args()

    pygame.init()
    names = args.track or list(TRACKS)
    built, skipped = [], []

    for name in names:
        try:
            meta = build_one(name, TRACKS[name])
        except RuntimeError as error:
            print(f"{name}: SKIPPED - {error}")
            meta = None
        if meta is None:
            skipped.append(name)
            continue
        built.append(meta)
        car = meta["car"]
        print(
            f"{name:14s} {meta['length_px']:5d}px lap  band {meta['median_width_px']:3d}px "
            f"({meta['width_in_cars']:.1f} cars)  car {car['size_x']}x{car['size_y']}px  "
            f"{len(meta['checkpoints']):2d} cps  {len(meta['apex_indices'])} apexes"
        )

    print(f"\n{len(built)} tracks built, {len(skipped)} skipped"
          + (f": {', '.join(skipped)}" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

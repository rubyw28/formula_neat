"""Download real circuit centrelines for the F1 calendar.

    python tools/fetch_circuits.py

Saves one GeoJSON per circuit into assets/tracks/geo/. Those files are committed,
so `tools/build_tracks.py` works offline and rebuilds are reproducible; you only
need to run this again when the calendar changes.

Source: https://github.com/bacinger/f1-circuits (MIT). Each file is a closed
LineString of the track centreline in WGS84, with the official lap length in its
properties. Measured against those stated lengths the geometry is accurate to
well under 1%.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO_DIR = os.path.join(ROOT, "assets", "tracks", "geo")
SOURCE = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits"

# 2026 Formula 1 calendar, in season order: (track name used by the sim,
# circuit id in the source dataset, human label).
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
    ("italy", "it-1922", "Autodromo Nazionale Monza"),
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


def fetch(circuit_id, destination, timeout):
    url = f"{SOURCE}/{circuit_id}.geojson"
    request = urllib.request.Request(url, headers={"User-Agent": "formula-neat"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    feature = payload["features"][0]
    geometry = feature["geometry"]
    coords = geometry["coordinates"]
    if geometry["type"] == "MultiLineString":
        coords = max(coords, key=len)
    if len(coords) < 32:
        raise ValueError(f"{circuit_id}: only {len(coords)} points")

    with open(destination, "w") as handle:
        json.dump(payload, handle)
    return len(coords), feature["properties"].get("length")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    os.makedirs(GEO_DIR, exist_ok=True)
    failures = []
    for name, circuit_id, label in CALENDAR:
        destination = os.path.join(GEO_DIR, f"{circuit_id}.geojson")
        if os.path.exists(destination) and not args.force:
            print(f"  have  {name:14s} {label}")
            continue
        try:
            points, length = fetch(circuit_id, destination, args.timeout)
            print(f"  got   {name:14s} {label}  ({points} points, {length}m)")
        except (urllib.error.URLError, ValueError, KeyError, OSError) as error:
            failures.append((name, error))
            print(f"  FAIL  {name:14s} {error}")

    print(f"\n{len(CALENDAR) - len(failures)}/{len(CALENDAR)} circuits in {GEO_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

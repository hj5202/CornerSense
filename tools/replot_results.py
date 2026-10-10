"""Redraw result figures from saved metric curves.

main reads result JSON and curve arrays, then updates PNG and SVG figures.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")

import numpy as np
from cornersense.plotting import plot_file, plot_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.results:
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document["results"]:
            if row["status"] != "ok":
                continue
            curve = np.load(path.parent / row["artifacts"]["curve"])
            plot_file(path.parent / row["artifacts"]["plot"], curve["time_s"], curve["Dphys_m"],
                      row["trace_extent_s"], row["evaluation"], row["filename"])
        plot_scene(path.parent / "overview.png", document["results"])
        print(f"Redrawn: {path}")


if __name__ == "__main__":
    main()

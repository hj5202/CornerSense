#!/usr/bin/env python3
"""Regenerate the aug-CSI power projection and CSI-power comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..core import constants
from .evaluate import compare_with_truth, extract
from .generate import generate
from .pipeline import detect
from .figures import make_figures


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False)+"\n")


def run(config_file, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    config = json.loads(config_file.read_text())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    observation, truth = generate(config, constants.FREQS, constants.C)
    detection = detect(observation["csi"], observation["time_s"], config["robot_speed_mps"])
    report, data, intermediate = extract(detection)
    report, data = compare_with_truth(report, data, intermediate, truth)
    np.savez_compressed(output / "observation.npz", **observation)
    np.savez_compressed(output / "truth.npz", **truth)
    np.savez_compressed(output / "stage1.npz", **intermediate)
    np.savez_compressed(output / "data.npz", **data)
    write_json(output / "report.json", report)
    make_figures(output)
    for method in ("augmented", "raw"):
        print(f"{method}: {report[f'{method}_vs_truth']}")
    print(f"Saved aug-CSI power projection comparisons to {output}")
    return report


def main(simulation_directory):
    """Run one project's configuration through the shared simulation tools."""
    directory = Path(simulation_directory)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=directory / "config.json")
    parser.add_argument("--output", type=Path, default=directory.parent / "results" / directory.name)
    args = parser.parse_args()
    run(args.config, args.output)

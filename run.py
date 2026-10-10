#!/usr/bin/env python3
"""Process CSI recordings and save metric curves, figures and result summaries."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
from cornersense.inference import infer, json_safe, load_calibration
from cornersense.evaluation import summarize_results
from cornersense.metric_units import metric_units, phase_evaluation, to_paper_g
from cornersense.plotting import plot_file, plot_scene


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_file(task, *, verbose=False):
    path, output, calibration = task
    path, output = Path(path), Path(output)
    folder = output / path.parent.name / path.name
    folder.mkdir(parents=True)
    row = {"group": path.parent.name, "filename": path.name, "status": "error"}
    failure = None
    with io.StringIO() as log, redirect_stdout(log):
        try:
            session = path.parent.name
            if session not in calibration:
                raise ValueError(f"No fixed AoA offset for session {session}")
            row, times, distance = infer(path, calibration[session])
            row["evaluation"] = phase_evaluation(row["evaluation"])
            tr = np.abs(distance)
            np.savez_compressed(folder / "curve.npz", time_s=times, Dphys_m=distance,
                                tr_m=tr, G_rad=to_paper_g(tr))
            plot_file(folder / "metrics.png", times, distance, row["trace_extent_s"], row["evaluation"], path.name)
            row["artifacts"] = {key: (folder / value).relative_to(output).as_posix() for key, value in
                                {"curve": "curve.npz", "plot": "metrics.png", "vector_plot": "metrics.svg"}.items()}
        except Exception as error:
            row.update(status="error", error=str(error))
            failure = traceback.format_exc()
        diagnostic = log.getvalue()
        write_json(folder / "result.json", row)
    if failure:
        print(f"{path.parent.name}/{path.name}: processing failed\n{diagnostic}{failure}",
              file=sys.stderr, flush=True)
    elif verbose:
        print(f"{path.parent.name}/{path.name}:\n{diagnostic}", end="", flush=True)
    else:
        for line in diagnostic.splitlines():
            if "WARNING:" in line:
                print(f"{path.parent.name}/{path.name}: {line.strip()}",
                      file=sys.stderr, flush=True)
    return row


def save_summary(output, rows, metadata):
    summary = summarize_results(rows)
    metadata = dict(metadata, metric_units=metric_units())
    write_json(output / "results.json", {"metadata": metadata, "summary": summary, "results": rows})
    fields = ["group", "filename", "status", "calibration_offset_deg", "predicted_motion", "Ig_m", "predicted_range",
              "tr_min_m", "first_half_tr_min_m", "second_half_tr_min_m",
              "G_min_rad", "first_half_G_min_rad", "second_half_G_min_rad",
              "motion_correct", "region_correct", "both_correct", "error"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ev = row.get("evaluation", {})
            halves = ev.get("half_tr_min_m", [])
            phase_halves = ev.get("half_G_min_rad", [])
            writer.writerow({"group": row["group"], "filename": row["filename"], "status": row["status"],
                "calibration_offset_deg": row.get("calibration_offset_deg"),
                "predicted_motion": ev.get("predicted_motion"), "Ig_m": ev.get("Ig"),
                "predicted_range": " -> ".join(ev.get("predicted_regions", [])), "tr_min_m": ev.get("tr_min_m"),
                "first_half_tr_min_m": halves[0] if halves else None,
                "second_half_tr_min_m": halves[1] if len(halves) == 2 else None,
                "G_min_rad": ev.get("G_min_rad"),
                "first_half_G_min_rad": phase_halves[0] if phase_halves else None,
                "second_half_G_min_rad": phase_halves[1] if len(phase_halves) == 2 else None,
                **{key: ev.get(key) for key in ("motion_correct", "region_correct", "both_correct")},
                "error": row.get("error", "")})
    plot_scene(output / "overview.png", rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description="CornerSense: raw CSI folder to direction and range results")
    parser.add_argument("--input", type=Path, required=True, help="Scene folder or parent containing scene folders")
    parser.add_argument("--output", type=Path, help="New result directory")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--verbose", action="store_true", help="Print detailed processing diagnostics")
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration/sample_angles.json")
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error("--input must be a scene directory")
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        data = load_calibration(args.calibration)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    files = sorted(p for p in args.input.resolve().rglob("*") if p.is_file() and
                   not any(part.startswith(".") for part in p.relative_to(args.input.resolve()).parts) and p.suffix == "")
    if not files:
        parser.error("No raw CSI recordings found")
    keys = [(p.parent.name, p.name) for p in files]
    if len(keys) != len(set(keys)):
        parser.error("Repeated scene/file names in input")
    missing = sorted({group for group, _ in keys} - set(data["offsets_deg"]))
    if missing:
        parser.error(f"No fixed AoA offset for session(s): {', '.join(missing)}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or ROOT / "results" / f"{args.input.name}_{run_id}").resolve()
    if output.exists():
        parser.error("--output must be a new directory")
    output.mkdir(parents=True)
    tasks = [(str(path), str(output), data["offsets_deg"]) for path in files]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, task, verbose=args.verbose): path
                   for task, path in zip(tasks, files)}
        for future in as_completed(futures):
            path = futures[future]
            row = future.result()
            results[(path.parent.name, path.name)] = row
            ev = row.get("evaluation", {})
            print(f"[{len(results)}/{len(files)}] {path.parent.name}/{path.name}: "
                  f"{ev.get('predicted_motion') if row['status'] == 'ok' else 'error'}  "
                  f"{' -> '.join(ev.get('predicted_regions', []))}"
                  f"{': ' + row['error'] if row.get('error') else ''}", flush=True)
    rows = [results[key] for key in keys]
    input_path = args.input.resolve()
    input_label = input_path.relative_to(ROOT).as_posix() if input_path.is_relative_to(ROOT) else input_path.name
    summary = save_summary(output, rows, {"created_utc": run_id, "input": input_label,
        "method": "CornerSense", "calibration": data,
        "workers": args.workers, "python": sys.version,
        "numpy": np.__version__})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output}")
    return 0 if summary["processing_ok"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

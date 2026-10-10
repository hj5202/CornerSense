"""Check sample coverage and recompute scores and calibrated angles from saved results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from cornersense.evaluation import evaluate_curve, protocol_label, summarize_results
from cornersense.inference import calibrate_angles, json_safe, load_calibration, validate_calibration
from cornersense.metric_units import metric_units, phase_evaluation, to_paper_g

SAMPLE_GROUPS = {"0319_data": 5, "0325_data": 5, "data_1013": 9, "data_sup01": 16}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_package():
    sample = ROOT / "sample"
    require({p.name for p in sample.iterdir() if p.is_dir()} == set(SAMPLE_GROUPS),
            "Sample session folders do not match the published set")
    recordings = set()
    for group, count in SAMPLE_GROUPS.items():
        files = [p for p in (sample / group).iterdir()
                 if p.is_file() and not p.name.startswith('.') and not p.suffix]
        require(len(files) == count, f"{group}: expected {count} recordings")
        require(all(p.is_file() and not p.suffix and p.stat().st_size > 0 for p in files),
                f"{group}: invalid or empty recording")
        recordings.update(p.relative_to(sample).as_posix() for p in files)
    calibration = load_calibration(ROOT / "calibration/sample_angles.json")["offsets_deg"]
    require(set(calibration) == set(SAMPLE_GROUPS),
            "Calibration sessions differ from the sample folders")
    return {"raw_files_verified": len(recordings),
            "calibration_records_verified": len(calibration)}


def compare(actual, expected, key):
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(np.isclose(actual, expected, rtol=1e-9, atol=1e-9),
                f"{key}: {actual!r} != {expected!r}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{key}: list lengths differ")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{key}[{i}]")
    else:
        require(actual == expected, f"{key}: {actual!r} != {expected!r}")


def verify_results(path):
    path = Path(path).resolve()
    document = read(path)
    require(document["metadata"]["metric_units"] == metric_units(),
            "Result metric units do not match the output convention")
    calibration = validate_calibration(document["metadata"].get("calibration"))["offsets_deg"]
    recomputed_rows = []
    seen = set()
    for row in document["results"]:
        identity = (row["group"], row["filename"])
        require(identity not in seen, f"Duplicate result: {identity}")
        seen.add(identity)
        require(row["status"] == "ok", f"Recording did not complete: {identity}")
        name = "/".join(identity)
        require((ROOT / "sample" / name).is_file(), f"Recording is missing: {name}")
        require(row["group"] in calibration, f"No saved calibration for session {row['group']}")
        offset = calibration[row["group"]]
        compare(row["calibration_offset_deg"], offset, f"{name}: calibration offset")
        angles = calibrate_angles(row["measured_aoa"], offset)
        for key, value in angles.items():
            compare(row["selected_aoa"][key], value, key)
        with np.load(path.parent / row["artifacts"]["curve"], allow_pickle=False) as curve:
            require(all(np.isfinite(curve[key]).all() for key in ("time_s", "Dphys_m", "tr_m", "G_rad")),
                    f"{name}: non-finite curve values")
            require(np.array_equal(curve["tr_m"], np.abs(curve["Dphys_m"])),
                    f"{name}: tr does not equal abs(Dphys)")
            require(np.array_equal(curve["G_rad"], to_paper_g(curve["tr_m"])),
                    f"{name}: G conversion differs")
            recomputed = phase_evaluation(evaluate_curve(
                protocol_label(row["filename"]), curve["Dphys_m"],
                curve["time_s"], *row["trace_extent_s"]))
        for key, value in recomputed.items():
            compare(row["evaluation"][key], value, key)
        require(read(path.parent / row["group"] / row["filename"] / "result.json") == row,
                f"{name}: per-recording result differs from the summary")
        for key in ("plot", "vector_plot"):
            require((path.parent / row["artifacts"][key]).stat().st_size > 1000,
                    f"{name}: {key} is empty or incomplete")
        recomputed_rows.append({"status": "ok", "evaluation": recomputed})
    counts = summarize_results(recomputed_rows)
    require(counts == document["summary"], "Recomputed counts differ from the saved summary")
    return {"result": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
            **counts, "saved_curves_recomputed": len(seen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "passed", "package": verify_package(),
              "runs": [verify_results(path.resolve()) for path in args.results]}
    text = json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

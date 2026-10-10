"""Estimate calibrated path angles and motion/range metrics from raw CSI.

CSI has shape (packets, 3 antennas, 30 tones). Measured and calibrated angles
are in degrees; infer converts them to radians for the processing pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .core import pipeline, spotfi
from .core.intel5300 import read_bf_file
from .evaluation import evaluate_curve, protocol_label


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


def decode(path):
    raw = read_bf_file(str(path), scale=True, remove_sm=False)
    h, t = raw.csi[:, 0], np.asarray(raw.timestamps_s, dtype=float)
    if len(t) < 2 or len(t) != len(h) or not np.all(np.isfinite(t)):
        raise ValueError("Invalid CSI timestamps")
    t = t - t[0]
    if t[-1] <= 0 or np.any(np.diff(t) < 0):
        raise ValueError("CSI timestamps must increase")
    return h, t


def measure_angles(h):
    spec, angles, delays, _ = spotfi.spotfi_music_spectrum(
        spotfi.sanitize_shared_packet_slope(h), source_count=2, frequency_aperture=6)
    keep = delays >= -25.
    spec, delays = spec[:, keep], delays[keep]
    spec = spec / spec.max()
    peaks = spotfi.select_peaks(spec, angles, delays)
    refs = [p for p in peaks if abs(p[0] - 30.) <= 25.]
    ref = max(refs or peaks, key=lambda p: p[2] - .03 * abs(p[0] - 30.))
    humans = [p for p in peaks if p is not ref and
              (abs(p[0] - ref[0]) >= 6. or abs(p[1] - ref[1]) >= 5.)]
    human = max(humans, key=lambda p: p[2]) if humans else ref
    return {"theta_r": float(ref[0]), "theta_u": float(human[0])}


def validate_calibration(document):
    """Validate a fixed session-offset configuration."""
    if not isinstance(document, dict) or document.get("method") != "fixed_session_offset":
        raise ValueError("Calibration method must be fixed_session_offset")
    offsets = document.get("offsets_deg")
    if not isinstance(offsets, dict) or not offsets:
        raise ValueError("Calibration requires offsets_deg keyed by session folder")
    for session, offset in offsets.items():
        if (not isinstance(session, str) or not session or session in (".", "..")
                or "/" in session or "\\" in session):
            raise ValueError("Calibration keys must be session folder names")
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not np.isfinite(offset):
            raise ValueError(f"Invalid AoA offset for session {session}")
    return document


def load_calibration(path):
    """Read fixed session offsets in degrees."""
    return validate_calibration(json.loads(Path(path).read_text(encoding="utf-8")))


def calibrate_angles(measured, offset_deg):
    """Subtract one session offset from both measured path angles."""
    if (isinstance(offset_deg, bool) or not isinstance(offset_deg, (int, float, np.number))
            or not np.isreal(offset_deg) or not np.isfinite(offset_deg)):
        raise ValueError("AoA offset must be a finite number in degrees")
    angles = {key: float(measured[key]) - float(offset_deg) for key in ("theta_r", "theta_u")}
    if any(not np.isfinite(value) or abs(value) > 80. for value in angles.values()):
        raise ValueError("Corrected AoA must lie within [-80, 80] degrees")
    return angles


def infer(path, offset_deg):
    path = Path(path)
    h, t = decode(path)
    measured = measure_angles(h)
    angles = calibrate_angles(measured, offset_deg)
    speed_match = re.search(r"0p([258])", path.name)
    speed = float("0." + speed_match[1]) if speed_match else 0.2
    core = pipeline.run_file(str(path),
        {"th_r": np.deg2rad(angles["theta_r"]), "th_u": np.deg2rad(angles["theta_u"]),
         "robot_speed": speed}, preloaded=(h, t))
    if not core or "Dphys" not in core:
        raise ValueError("No metric curve was produced")
    evaluation = evaluate_curve(protocol_label(path.name), core["Dphys"],
                                core["metric_t"], float(t[0]), float(t[-1]),
                                split_mode="trace_time")
    if evaluation["gt_motion"] == "no_person":
        range_text = "not applicable"
    elif evaluation["predicted_regions"]:
        range_text = " -> ".join(evaluation["predicted_regions"])
    elif evaluation.get("range_status") == "missing-physical-half-coverage":
        range_text = "unavailable (insufficient coverage of a trace half)"
    else:
        range_text = "unavailable (unrecognized recording label)"
    print(f"    evaluation: motion={evaluation['predicted_motion']}; range={range_text}")
    record = {"filename": path.name, "group": path.parent.name, "status": "ok",
              "robot_speed_mps": speed, "measured_aoa": measured, "selected_aoa": angles,
              "calibration_offset_deg": float(offset_deg),
              "trace_extent_s": [float(t[0]), float(t[-1])], "packets": len(t),
              "evaluation": evaluation,
              "diagnostics": {k: core[k] for k in ("seg", "fs", "selected_key_pair",
                  "selected_key_frequency_hz") if k in core}}
    return json_safe(record), np.asarray(core["metric_t"]), np.asarray(core["Dphys"])

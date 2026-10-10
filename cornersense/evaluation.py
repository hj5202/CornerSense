"""Evaluate motion and range from metric curves on the recording timeline."""
import numpy as np
from .labels import label_for


def protocol_label(path):
    return label_for(path)


def summarize_results(rows):
    """Count evaluated recordings, keeping no-person range results separate."""
    rows = list(rows)
    evaluations = [row["evaluation"] for row in rows if row.get("status") == "ok"]
    summary = {"total": len(rows), "processing_ok": len(evaluations)}
    for field, denominator in (("motion_correct", "motion_total"),
                               ("region_correct", "range_total"),
                               ("both_correct", "both_total")):
        summary[denominator] = sum(ev.get(field) is not None for ev in evaluations)
        summary[field] = sum(ev.get(field) is True for ev in evaluations)
    no_person = [ev for ev in evaluations if ev.get("gt_motion") == "no_person"]
    summary["noperson_total"] = len(no_person)
    summary["noperson_correct"] = sum(ev["motion_correct"] is True for ev in no_person)
    return summary


def evaluate_curve(label, distance, metric_t, trace_start, trace_end,
                   split_mode="trace_time"):
    d = np.asarray(distance, dtype=float)
    t = np.asarray(metric_t, dtype=float)
    if len(d) < 6 or d.shape != t.shape or not np.all(np.isfinite(d)):
        raise ValueError("invalid metric curve")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite([trace_start,trace_end])):
        raise ValueError("timestamps and trace bounds must be finite")
    if trace_end <= trace_start:
        raise ValueError("trace end must follow trace start")
    if np.any(np.diff(t) < 0):
        raise ValueError("metric timestamps must be monotonic")
    tr = np.abs(d)
    k = max(3, len(tr) // 5)
    ig = float(np.median(tr[-k:]) - np.median(tr[:k]))
    motion = "approaching" if ig > .10 else "moving_away" if ig < -.10 else "noperson"
    gt_motion = label.get("motion")
    no_person = gt_motion == "no_person"
    motion_correct = None
    if gt_motion in ("approaching", "moving_away", "no_person"):
        motion_correct = motion == ("noperson" if no_person else gt_motion)
    result = dict(Ig=ig, tr_min_m=float(tr.min()), predicted_motion=motion,
                  motion_correct=motion_correct, gt_motion=gt_motion,
                  gt_region=None if no_person else label.get("region"),
                  range_eval=split_mode, half_tr_min_m=[], predicted_regions=[],
                  region_correct=None, both_correct=None, status="ok")
    if no_person:
        return result
    if label.get("region_origin") == "temporal_half":
        midpoint = (float(trace_start) + float(trace_end)) / 2
        if split_mode == "trace_time":
            masks = [t < midpoint, t >= midpoint]
        elif split_mode == "metric_packet":
            masks = [np.arange(len(t)) < len(t)//2,
                     np.arange(len(t)) >= len(t)//2]
        else:
            raise ValueError(split_mode)
        result["raw_time_midpoint_s"] = midpoint
        result["half_metric_counts"] = [int(m.sum()) for m in masks]
        result["metric_time_extent_s"] = [float(t[0]), float(t[-1])]
        result["trace_time_extent_s"] = [float(trace_start), float(trace_end)]
        result["half_observed_span_s"] = [
            float(np.ptp(t[m])) if m.sum() else 0. for m in masks]
        if any(m.sum() < 3 for m in masks):
            result["range_status"] = "missing-physical-half-coverage"
            return result
        mins = [float(tr[m].min()) for m in masks]
        regions = ["R1" if x > .85 else "R2" for x in mins]
        result.update(half_tr_min_m=mins, predicted_regions=regions)
        if gt_motion in ("approaching", "moving_away"):
            expected = ["R2", "R1"] if gt_motion == "approaching" else ["R1", "R2"]
            result.update(expected_regions=expected, region_correct=regions == expected)
    elif label.get("region") in ("R1", "R2"):
        region = "R1" if tr.min() > .85 else "R2"
        result.update(predicted_regions=[region],
                      region_correct=region == label["region"])
    if motion_correct is not None and result["region_correct"] is not None:
        result["both_correct"] = bool(result["motion_correct"] and
                                        result["region_correct"])
    return result

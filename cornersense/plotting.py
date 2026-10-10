"""Create static direction and range figures from CornerSense predictions.

plot_file plots one metric curve and its range decision; plot_scene
compares direction scores and range minima for every recording in a scene.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metric_units import G_THRESHOLD_RAD, to_paper_g


GREEN = "#16826A"
BLUE = "#3677B5"
GRAY = "#747C83"
MOTION_NAMES = {
    "approaching": "Approaching",
    "moving_away": "Moving away",
    "noperson": "noperson",
}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#30383F",
    "axes.titlecolor": "#24323C",
    "xtick.color": "#4E5963",
    "ytick.color": "#4E5963",
    "savefig.facecolor": "white",
}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _motion_color(motion):
    return GREEN if motion == "approaching" else BLUE if motion == "moving_away" else GRAY


def _range_values(evaluation):
    halves = evaluation.get("half_tr_min_m") or []
    if halves:
        return [_number(value) for value in halves]
    if evaluation.get("raw_time_midpoint_s") is not None:
        return []
    minimum = _number(evaluation.get("tr_min_m"))
    return [] if minimum is None else [minimum]


def _range_result(evaluation):
    if evaluation.get("gt_motion") == "no_person":
        return [], [], "not applicable"
    if evaluation.get("range_status") == "missing-physical-half-coverage":
        return [], [], "unavailable (insufficient coverage of a trace half)"
    regions = evaluation.get("predicted_regions") or []
    if not regions and evaluation.get("gt_motion") in (None, "unknown"):
        return [], [], "unavailable (unrecognized recording label)"
    minima = _range_values(evaluation)
    count = 2 if evaluation.get("raw_time_midpoint_s") is not None else 1
    if (len(regions) != count or len(minima) != count
            or any(region not in ("R1", "R2") for region in regions)
            or any(value is None for value in minima)):
        return [], [], "unavailable (missing or inconsistent range results)"
    return minima, regions, None


def _range_text(evaluation):
    minima, regions, reason = _range_result(evaluation)
    if reason is not None:
        return f"Range: {reason}"
    if len(regions) == 2:
        return (f"Range: first half {regions[0]} (min tr={minima[0]:.3f} m); "
                f"second half {regions[1]} (min tr={minima[1]:.3f} m)")
    return f"Range: {regions[0]} (min tr={minima[0]:.3f} m)"


def _save(fig, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path.with_suffix(".png"), dpi=160, format="png", bbox_inches="tight")
        fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def plot_file(path, times, distance, trace_extent, evaluation, filename):
    times = np.asarray(times, dtype=float)
    tr = np.abs(np.asarray(distance, dtype=float))
    if (times.ndim != 1 or tr.shape != times.shape or len(times) < 6
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(tr))):
        raise ValueError("Plot requires matching finite time and metric arrays.")
    if np.any(np.diff(times) < 0):
        raise ValueError("Metric timestamps must be monotonic.")
    count = max(3, len(tr) // 5)
    first_tr = float(np.median(tr[:count]))
    last_tr = float(np.median(tr[-count:]))
    score = _number(evaluation.get("Ig"))
    if score is None:
        score = last_tr - first_tr
    paper_g = to_paper_g(tr)
    first = float(to_paper_g(first_tr))
    last = float(to_paper_g(last_tr))
    motion = evaluation.get("predicted_motion", "unavailable")
    color = _motion_color(motion)
    has_range = _range_result(evaluation)[2] is None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.2, 5.6))
        fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.28)
        fig.suptitle(str(filename), fontsize=13, y=0.98)
        ax.set_title(f"{MOTION_NAMES.get(motion, 'Unavailable')}  |  Ig = {score:+.3f} m",
                     fontsize=12, pad=12, color=color)
        ax.plot(times, paper_g, color=color, linewidth=1.4, label="G(t) = (2πΔf/c) tr(t)")
        ax.plot([times[0], times[count - 1]], [first, first],
                color="#24323C", linewidth=2.4, label="First / last 20% medians")
        ax.plot([times[-count], times[-1]], [last, last],
                color="#24323C", linewidth=2.4)
        if has_range:
            ax.axhline(G_THRESHOLD_RAD, color=GRAY, linewidth=1.2, linestyle="--",
                       label="T_R = 0.85 m (converted to rad)")
        midpoint = _number(evaluation.get("raw_time_midpoint_s"))
        if midpoint is not None:
            ax.axvline(midpoint, color=GRAY, linewidth=1.2, linestyle=":",
                       label="Raw-trace midpoint")
        extent = trace_extent
        if extent and len(extent) == 2:
            start, end = map(_number, extent)
            if start is not None and end is not None and end > start:
                ax.set_xlim(start, end)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("G(t) (rad)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.20, linewidth=0.7)
        ax.legend(loc="best", ncol=1, frameon=True, framealpha=0.9, fontsize=9)
        fig.text(0.09, 0.105, _range_text(evaluation), fontsize=11, color="#24323C")
        fig.text(0.09, 0.06,
                 "Ig = median(tr, last 20%) - median(tr, first 20%); approaching > +0.10 m, moving away < -0.10 m.",
                 fontsize=9, color="#4E5963")
        if has_range:
            fig.text(0.09, 0.025, "Range decision: R1 if min tr > T_R = 0.85 m; R2 otherwise.",
                     fontsize=9, color="#4E5963")
        return _save(fig, path)


def plot_scene(path, rows):
    records = list(rows)
    if not records:
        raise ValueError("A scene plot requires at least one recording.")
    names = [str(row.get("filename", "recording")) for row in records]
    positions = np.arange(len(records))
    success = sum(row.get("status") == "ok" for row in records)
    groups = sorted({str(row.get("group", "")) for row in records} - {""})
    heading = ", ".join(groups) if groups else "Scene"
    height = max(4.6, 2.3 + 0.47 * len(records))
    with plt.rc_context(STYLE):
        fig, (direction_ax, range_ax) = plt.subplots(
            1, 2, figsize=(13.5, height), sharey=True, gridspec_kw={"wspace": 0.27})
        fig.subplots_adjust(left=0.255, right=0.96, top=0.83, bottom=0.17)
        fig.suptitle(f"{heading}  |  {success}/{len(records)} recordings processed",
                     fontsize=14, y=0.975)
        max_score = 0.15
        max_range = float(to_paper_g(1.0))
        has_range = False
        for position, row in zip(positions, records):
            evaluation = row.get("evaluation") or {}
            score = _number(evaluation.get("Ig"))
            if row.get("status") != "ok" or score is None:
                for axis in (direction_ax, range_ax):
                    axis.text(0.02, position, "Unavailable", color=GRAY, va="center",
                              transform=axis.get_yaxis_transform(), fontsize=9)
                continue
            max_score = max(max_score, abs(score))
            direction_ax.barh(position, score, height=0.55,
                              color=_motion_color(evaluation.get("predicted_motion")))
            direction_ax.annotate(f"{score:+.3f}", (score, position),
                                  xytext=(4 if score >= 0 else -4, 0),
                                  textcoords="offset points", va="center",
                                  ha="left" if score >= 0 else "right", fontsize=9)
            minima, regions, reason = _range_result(evaluation)
            if reason is not None:
                range_ax.text(0.02, position, reason.capitalize(), color=GRAY,
                              transform=range_ax.get_yaxis_transform(), va="center", fontsize=8)
                continue
            has_range = True
            for index, value in enumerate(minima):
                offset = 0 if len(minima) == 1 else (-0.17 if index == 0 else 0.17)
                region = regions[index]
                label = region if len(minima) == 1 else f"H{index + 1}: {region}"
                paper_g = float(to_paper_g(value))
                range_ax.barh(position + offset, paper_g,
                              height=0.52 if len(minima) == 1 else 0.29,
                              color=GREEN if region == "R1" else BLUE)
                range_ax.annotate(f"{paper_g:.4f}  {label}", (paper_g, position + offset),
                                  xytext=(4, 0), textcoords="offset points",
                                  va="center", fontsize=8.5)
                max_range = max(max_range, paper_g)
        direction_ax.set_yticks(positions, labels=names, fontsize=9)
        direction_ax.set_ylim(len(records) - 0.4, -0.6)
        direction_ax.set_xlim(-max_score * 1.4, max_score * 1.4)
        direction_ax.axvline(0, color=GRAY, linewidth=0.8)
        for boundary in (-0.1, 0.1):
            direction_ax.axvline(boundary, color=GRAY, linewidth=1.0, linestyle="--")
        direction_ax.set_title("Direction", pad=12)
        direction_ax.set_xlabel("Ig (m)  (away < -0.10; approaching > +0.10)", fontsize=10)
        if has_range:
            range_ax.axvline(G_THRESHOLD_RAD, color=GRAY, linewidth=1.0, linestyle="--")
        range_ax.set_xlim(0, max_range * 1.45)
        range_ax.set_title("Range", pad=12)
        range_ax.set_xlabel("Minimum G (rad); threshold from T_R = 0.85 m"
                            if has_range else "Minimum G (rad)", fontsize=10)
        range_ax.tick_params(axis="y", left=False)
        for axis in (direction_ax, range_ax):
            axis.set_axisbelow(True)
            axis.grid(axis="x", alpha=0.15, linewidth=0.7)
            axis.spines["left"].set_visible(False)
        fig.text(0.255, 0.05,
                 "H1 / H2: first / second half of the raw trace.  G = (2πΔf/c) tr.",
                 fontsize=9, color="#4E5963")
        return _save(fig, path)

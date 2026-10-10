"""Plot the aug-CSI power projection comparison and uncentered CSI powers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2563A6"
ORANGE = "#C96726"
GREEN = "#3D8060"
GRAY = "#59616B"

DATA_FIELDS = (
    "packet_index",
    "time_s",
    "raw_power",
    "augmented_power",
    "truth_cos",
    "augmented_projection_display",
    "raw_projection_display",
)


def _read_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        data = {name: np.asarray(saved[name]) for name in DATA_FIELDS}
    length = len(data["packet_index"])
    if length < 2:
        raise ValueError("At least two packets are required to plot a run.")
    for name, values in data.items():
        if values.ndim != 1 or len(values) != length:
            raise ValueError(f"{name} must be a one-dimensional, packet-aligned array.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
    if np.any(np.diff(data["packet_index"]) <= 0):
        raise ValueError("packet_index must be strictly increasing.")
    return data


def _style_axes(ax, packets):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", color="#D4D8DD", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_xlim(float(packets[0]), float(packets[-1]))
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.margins(y=0.09)


def _save(fig, directory: Path, name: str) -> list[Path]:
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _correlation_note(report):
    labels = []
    for key, label in (("augmented_vs_truth", "aug-CSI"),
                       ("raw_vs_truth", "Raw")):
        score = report.get(key, {})
        correlation = score.get("correlation")
        rmse = score.get("rmse")
        if correlation is not None and rmse is not None:
            labels.append(f"{label}: r = {correlation:.3f}, RMSE = {rmse:.3f}")
    return "     |     ".join(labels)


def _cosine_figure(data, report, directory):
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.79)
    packet = data["packet_index"]
    truth, = ax.plot(
        packet, data["truth_cos"], color=GREEN, linewidth=1.5,
        label="Reference–virtual cosine (truth)", zorder=2,
    )
    augmented, = ax.plot(
        packet, data["augmented_projection_display"], color=BLUE,
        linewidth=1.6, linestyle="--",
        label="aug-CSI power projection", zorder=4,
    )
    raw, = ax.plot(
        packet, data["raw_projection_display"], color=ORANGE,
        linewidth=1.7, linestyle=":",
        label="PCA component (raw power)", zorder=3,
    )
    _style_axes(ax, packet)
    ax.set_xlabel("Packet")
    ax.set_ylabel("Normalized component")
    fig.suptitle("Cosine-phase comparison", fontsize=12, y=0.985)
    fig.legend(
        handles=[augmented, raw, truth], frameon=False, ncol=1,
        loc="upper left", bbox_to_anchor=(0.10, 0.942),
        borderaxespad=0, handlelength=3.0, labelspacing=0.32, fontsize=8.7,
    )
    note = _correlation_note(report)
    if note:
        fig.text(0.10, 0.064, note, fontsize=8.5, color=GRAY, ha="left")
    fig.text(
        0.10, 0.020,
        "PCA projections are centered and scaled by their own √2 × RMS.",
        fontsize=8.2, color=GRAY, ha="left",
    )
    return _save(fig, directory, "cosine_comparison")


def _power_figure(data, directory):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9.6, 5.4))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.13, top=0.92, hspace=0.35)
    packet = data["packet_index"]
    series = (
        ("raw_power", "Raw CSI power", ORANGE),
        ("augmented_power", "Virtual-path augmented CSI power", BLUE),
    )
    for ax, (key, title, color) in zip(axes, series):
        ax.plot(packet, data[key], color=color, linewidth=1.3)
        _style_axes(ax, packet)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Power (a.u.)")
        ax.set_title(title, loc="left", fontsize=10.5, pad=7)
    axes[-1].set_xlabel("Packet")
    fig.text(
        0.10, 0.020, "Uncentered power; each panel uses its own vertical scale.",
        fontsize=8.2, color=GRAY, ha="left",
    )
    return _save(fig, directory, "power_comparison")


def make_figures(output: Path) -> list[Path]:
    """Render the PNG/SVG figures from a saved aug-CSI power projection run."""
    output = Path(output)
    data = _read_data(output / "data.npz")
    report = json.loads((output / "report.json").read_text())
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "svg.fonttype": "none",
    }):
        return (_cosine_figure(data, report, directory)
                + _power_figure(data, directory))


def main():
    """Redraw the figures in a saved simulation output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Saved aug-CSI power projection run directory")
    arguments = parser.parse_args()
    for figure in make_figures(arguments.output):
        print(figure)


if __name__ == "__main__":
    main()

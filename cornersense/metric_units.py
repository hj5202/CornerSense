"""Convert the metre-based tr metric to the paper's phase metric for output.

to_paper_g scales curves; phase_evaluation adds phase-valued score summaries;
metric_units describes the shared output conversion and range threshold.
"""
import numpy as np

from .core.pipeline import C, DF

PHASE_SCALE_RAD_PER_M = 2 * np.pi * DF / C
G_THRESHOLD_RAD = .85 * PHASE_SCALE_RAD_PER_M


def to_paper_g(tr):
    return np.asarray(tr, dtype=float) * PHASE_SCALE_RAD_PER_M


def phase_evaluation(evaluation):
    result = dict(evaluation)
    if "tr_min_m" in result:
        result["G_min_rad"] = float(to_paper_g(result["tr_min_m"]))
    if "half_tr_min_m" in result:
        result["half_G_min_rad"] = to_paper_g(result["half_tr_min_m"]).tolist()
    if "truth_on_metric_support" in result:
        result["truth_on_metric_support"] = phase_evaluation(result["truth_on_metric_support"])
    return result


def metric_units():
    return dict(tr="m", G="rad", Ig="m", phase_spacing_hz=DF,
                propagation_speed_mps=C, phase_scale_rad_per_m=PHASE_SCALE_RAD_PER_M,
                T_R_m=.85, G_threshold_rad=G_THRESHOLD_RAD)

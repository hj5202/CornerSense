"""Estimate subcarrier phase slope from weighted adjacent differences."""
from __future__ import annotations

import numpy as np


def _vector(value, name, *, complex_ok=False):
    a = np.asarray(value, dtype=complex if complex_ok else float)
    if a.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return a


def _frequencies(value):
    x = _vector(value, 'frequencies_hz')
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('frequencies_hz must be finite and strictly increasing')
    return x


def weighted_adjacent_slope(frequencies_hz, phase_unwrapped, weights=None):
    x = _frequencies(frequencies_hz)
    y = _vector(phase_unwrapped, 'phase_unwrapped')
    w = np.ones(len(x)) if weights is None else _vector(weights, 'weights')
    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError('frequency, phase and weight lengths must match')
    if np.any(w[np.isfinite(w)] < 0):
        raise ValueError('weights must be nonnegative')
    idx = np.flatnonzero(np.isfinite(y) & np.isfinite(w) & (w > 0))
    if len(idx) < 2:
        return dict(status='insufficient_carriers', slope=np.nan,
                    edge_indices=np.empty((0, 2), dtype=int),
                    edge_weights=np.array([]), quality=0.0, n_valid=len(idx))
    xv, yv, wv = x[idx], y[idx], w[idx]
    q = (wv / np.max(wv)) ** 2
    q /= q.sum()

    dx = xv - xv[0]
    xc = dx - np.sum(q * dx)
    denom = np.sum(q * xc ** 2)
    if not np.isfinite(denom) or denom <= np.finfo(float).tiny:
        return dict(status='degenerate_weights', slope=np.nan,
                    edge_indices=np.empty((0, 2), dtype=int),
                    edge_weights=np.array([]), quality=0.0, n_valid=len(idx))
    coefficients = q * xc / denom

    edge_coefficients = np.cumsum(coefficients[:0:-1])[::-1]
    gamma = edge_coefficients * np.diff(xv)
    local_slopes = np.diff(yv) / np.diff(xv)
    slope = float(gamma @ local_slopes)
    mean_phase = np.sum(q * yv)
    residual = yv - (mean_phase + slope * xc)
    rms = float(np.sqrt(np.sum(q * residual ** 2)))
    return dict(status='ok', slope=slope,
                intercept_at_first_hz=float(mean_phase - slope * np.sum(q * dx)),
                frequency_origin_hz=float(xv[0]),
                edge_indices=np.column_stack((idx[:-1], idx[1:])),
                edge_weights=gamma, adjacent_slopes=local_slopes,
                residual_rms_rad=rms, quality=float(np.exp(-rms ** 2)),
                n_valid=len(idx), has_bridged_edges=bool(np.any(np.diff(idx) > 1)))

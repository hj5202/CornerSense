"""SpotFi angle estimation with CSI phase preprocessing and MUSIC peak selection.

sanitize_shared_packet_slope removes packet phase slope; spatial_frequency_snapshots
forms smoothed observations; spotfi_music_spectrum computes the angle-delay
spectrum; select_peaks returns separated angle-delay candidates.
"""
from functools import lru_cache
import numpy as np
from .constants import C, FC, SUB, DF

ANTENNA_SPACING = 0.025
NUM_ANTENNAS = 3
SUBCARRIER_STRIDE = 2
SMOOTHED_DF = SUBCARRIER_STRIDE * DF


def sanitize_shared_packet_slope(csi):
    values = np.asarray(csi, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1] != NUM_ANTENNAS:
        raise ValueError("CSI must have shape (packets, 3, subcarriers)")
    freqs = FC + SUB * DF
    adjacent = values[:, :, 1:] * np.conj(values[:, :, :-1])
    slopes = np.angle(adjacent) / np.diff(freqs)[None, None, :]
    common = np.median(slopes, axis=(1, 2))
    correction = np.exp(-1j * common[:, None, None] *
                        (freqs - FC)[None, None, :])
    return values * correction


def select_peaks(spectrum, angles, delays, count=12):
    candidates = []
    for ai in range(1, len(angles) - 1):
        for di in range(1, len(delays) - 1):
            value = float(spectrum[ai, di])
            if value >= float(np.max(spectrum[ai - 1:ai + 2, di - 1:di + 2])):
                candidates.append((value, ai, di))
    flat = np.argpartition(spectrum.ravel(), -min(100, spectrum.size))[-min(100, spectrum.size):]
    for index in flat:
        ai, di = np.unravel_index(int(index), spectrum.shape)
        candidates.append((float(spectrum[ai, di]), ai, di))
    selected = []
    for value, ai, di in sorted(candidates, reverse=True):
        angle, delay = float(angles[ai]), float(delays[di])
        if any(abs(angle - p[0]) < 6.0 and abs(delay - p[1]) < 5.0
               for p in selected):
            continue
        selected.append((angle, delay, value))
        if len(selected) >= count:
            break
    return selected


def _validate(csi):
    values = np.asarray(csi, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1:] != (NUM_ANTENNAS, len(SUB)):
        raise ValueError("CSI must have shape (packets, 3, 30)")
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("CSI must be nonempty and finite")
    return values


def _uniform_runs(minimum_length):
    breaks = np.flatnonzero(np.diff(SUB) != SUBCARRIER_STRIDE) + 1
    return [run for run in np.split(np.arange(len(SUB)), breaks)
            if len(run) >= minimum_length]


def spatial_frequency_snapshots(csi, frequency_aperture=6,
                                spatial_aperture=2, max_packets=80):
    values = _validate(csi)
    if not 1 <= spatial_aperture <= NUM_ANTENNAS:
        raise ValueError("Invalid spatial aperture")
    if frequency_aperture < 2 or max_packets < 1:
        raise ValueError("Invalid frequency aperture or packet budget")
    runs = _uniform_runs(frequency_aperture)
    if not runs:
        raise ValueError("No uniform subcarrier run fits the aperture")
    packets = np.unique(np.linspace(0, len(values) - 1,
                                   min(max_packets, len(values))).astype(int))
    snapshots = []
    for packet in packets:
        for run in runs:
            for start in range(len(run) - frequency_aperture + 1):
                carriers = run[start:start + frequency_aperture]
                for rx in range(NUM_ANTENNAS - spatial_aperture + 1):
                    vector = values[packet, rx:rx + spatial_aperture][:, carriers].reshape(-1)
                    norm = np.linalg.norm(vector)
                    if norm > 1e-12:
                        snapshots.append(vector / norm)
    if len(snapshots) < 20:
        raise RuntimeError("Too few nonzero smoothing snapshots")
    return np.asarray(snapshots)


@lru_cache(maxsize=8)
def _steering_dictionary(frequency_aperture, spatial_aperture,
                         angle_step=2.0, delay_step_ns=2.5):
    angles = np.arange(-80.0, 80.0 + angle_step / 2, angle_step)
    delays = np.arange(-75.0, 75.0 + delay_step_ns / 2, delay_step_ns)
    spatial = np.exp(1j * 2 * np.pi * ANTENNA_SPACING * FC / C *
                     np.sin(np.deg2rad(angles))[:, None] *
                     np.arange(spatial_aperture)[None, :])
    frequency = np.exp(-1j * 2 * np.pi * SMOOTHED_DF *
                       delays[:, None] * 1e-9 *
                       np.arange(frequency_aperture)[None, :])
    dictionary = np.einsum("ar,df->adrf", spatial, frequency).reshape(
        len(angles) * len(delays), spatial_aperture * frequency_aperture)
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    return angles, delays, dictionary


def _model_order(eigenvalues):
    eig = np.maximum(np.asarray(eigenvalues).real, 0.0)
    noise_modes = eig[:-2] if len(eig) > 2 else eig[:1]
    floor = max(float(np.median(noise_modes)), float(eig[-1]) * 1e-10, 1e-14)
    ratio = float(eig[-2] / floor)
    relative = float(eig[-2] / max(eig[-1], 1e-14))
    return (2 if len(eig) > 2 and ratio >= 4.0 and relative >= 1e-3 else 1), ratio, relative, floor


def spotfi_music_spectrum(csi, source_count=None, frequency_aperture=6,
                          spatial_aperture=2, max_packets=80):
    snapshots = spatial_frequency_snapshots(
        csi, frequency_aperture, spatial_aperture, max_packets)
    covariance = snapshots.T @ snapshots.conj() / len(snapshots)
    covariance = (covariance + covariance.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    inferred, ratio, relative, floor = _model_order(eigenvalues)
    selected = inferred if source_count is None else int(source_count)
    if not 1 <= selected < covariance.shape[0]:
        raise ValueError("source_count must leave a nonempty noise subspace")
    noise = eigenvectors[:, :covariance.shape[0] - selected]
    angles, delays, dictionary = _steering_dictionary(
        frequency_aperture, spatial_aperture)
    denominator = np.sum(np.abs(dictionary.conj() @ noise) ** 2, axis=1)
    spectrum = (1.0 / np.maximum(denominator, 1e-10)).reshape(len(angles), len(delays))
    spectrum /= spectrum.max()
    diagnostics = {
        "snapshot_count": int(len(snapshots)),
        "frequency_aperture": int(frequency_aperture),
        "spatial_aperture": int(spatial_aperture),
        "frequency_spacing_hz": float(SMOOTHED_DF),
        "inferred_source_count": inferred, "music_source_count": selected,
        "signal_to_noise_eigenvalue_ratio": ratio,
        "second_to_first_eigenvalue_ratio": relative,
        "first_to_noise_eigenvalue_ratio": float(eigenvalues[-1].real / floor),
        "noise_eigenvalue_floor": floor,
        "eigenvalues": eigenvalues.real.tolist(),
        "steering_convention": "positive_spatial_negative_delay",
    }
    return spectrum, angles, delays, diagnostics

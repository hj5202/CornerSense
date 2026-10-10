"""Compare the CornerSense aug-CSI power projection with a raw-power control."""
import numpy as np

from ..core import pipeline

TARGET_TONE = 14
SMOOTHING_SECONDS = 0.2


def display_scale(values):
    """Normalize a cosine-shaped component with its own centered RMS."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    rms = np.sqrt(np.mean(centered**2))
    return centered / (np.sqrt(2) * rms) if rms > 1e-14 else np.zeros_like(centered)


def metrics(estimate, truth):
    x, y = np.asarray(estimate), np.asarray(truth)
    if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Comparison requires equal-shaped finite arrays")
    corr = float(np.corrcoef(x, y)[0, 1]) if min(x.std(), y.std()) > 1e-14 else None
    return {"correlation": corr, "rmse": float(np.sqrt(np.mean((x-y)**2)))}


def dominant_frequency(values, fs):
    """Descriptive spectrum statistic; not used for component extraction."""
    centered = values - np.mean(values)
    amplitude = np.abs(np.fft.rfft(centered * np.hanning(len(values))))
    return float(np.fft.rfftfreq(len(values), 1/fs)[int(np.argmax(amplitude))])


def extract(detection):
    """Extract augmented-CSI and raw-power PCA projections from returned intermediates."""
    result = detection["result"]
    if "intermediates" not in result:
        raise ValueError(f"CornerSense did not export aug-CSI power projection intermediates: {result.get('status')}")
    mid, fs = result["intermediates"], result["fs"]
    raw_power = np.abs(mid["Hs"][:, 0, :])**2
    augmented_power = np.abs(mid["Hs"][:, 0, :] + mid["Ha"])**2
    raw_processed = pipeline.bandlimit_t(pipeline.ma2d(raw_power, fs, SMOOTHING_SECONDS),
        fs, pipeline.BAND_LO, pipeline.BAND_HI).real
    raw_score, raw_loading, raw_singular = pipeline.stage_pca(raw_processed)
    stage1 = mid["stage1"]
    aug_projection = stage1["score"] * stage1["loading"][TARGET_TONE]
    raw_projection = raw_score * raw_loading[TARGET_TONE]
    st, en = map(int, result["seg"])
    data = {"time_s": np.asarray(result["metric_t"]),
            "packet_index": np.arange(st, en),
            "augmented_projection_display": display_scale(aug_projection),
            "raw_projection_display": display_scale(raw_projection),
            "raw_power": raw_power[:, TARGET_TONE],
            "augmented_power": augmented_power[:, TARGET_TONE]}
    intermediate = {"time_s": data["time_s"], "Hs": mid["Hs"], "Ha": np.asarray(mid["Ha"]),
        "augmented_power_matrix": augmented_power, "augmented_processed": mid["P2p"],
        "augmented_score": stage1["score"], "augmented_loading": stage1["loading"],
        "augmented_singular_values": stage1["singular_values"],
        "augmented_projection": aug_projection, "raw_power_matrix": raw_power,
        "raw_processed": raw_processed, "raw_score": raw_score,
        "raw_loading": raw_loading, "raw_singular_values": raw_singular,
        "raw_projection": raw_projection}
    index, lock, phase, sign = stage1["match"]
    report = {"segment_samples": [st, en], "sample_rate_hz": fs,
        "stage1_beat_match": {"frequency_hz": mid["tracked"][index]["f"] if index is not None else None,
                              "phase_lock": lock, "phase_offset_rad": phase, "sign": sign},
        "dominant_frequency_hz": {"augmented_projection": dominant_frequency(aug_projection, fs),
                                  "raw_projection": dominant_frequency(raw_projection, fs)},
        "processing_settings": {"band_hz": [pipeline.BAND_LO, pipeline.BAND_HI],
                                "smoothing_s": SMOOTHING_SECONDS},
        "measured_aoa_deg": detection["measured_aoa_deg"],
        "virtual_amplitude": float(abs(mid["Ha"])), "virtual_phase_rad": float(np.angle(mid["Ha"])),
        "target_antenna_one_based": 1, "target_tone_one_based": TARGET_TONE+1,
        "display_definition": "score * loading[tone], centered and divided by its own sqrt(2)*RMS; sign and phase unchanged",
        "raw_control_definition": "Same CSI segment and CornerSense ma2d, bandlimit_t, stage_pca applied to raw power"}
    return report, data, intermediate


def compare_with_truth(report, data, intermediate, truth):
    """Attach the synthesis comparison only after both projections are extracted."""
    st, en = report["segment_samples"]
    if not np.array_equal(data["time_s"], truth["time_s"][st:en]):
        raise ValueError("The truth and processing timestamps differ")
    reference = list(truth["path_names"]).index("reference")
    field = truth["path_fields"][reference, st:en, 0, TARGET_TONE]
    data["truth_cos"] = np.cos(np.angle(field) - np.angle(intermediate["Ha"]))
    core = slice(len(field)//5, len(field)-len(field)//5)
    for method in ("augmented", "raw"):
        trace = data[f"{method}_projection_display"]
        report[f"{method}_vs_truth"] = metrics(trace, data["truth_cos"])
        report[f"{method}_central_60_percent"] = metrics(trace[core], data["truth_cos"][core])
    report["truth_definition"] = "cos(arg(H_reference)-arg(Ha)) = cos(phi_reference+phi_a), with H_reference=A*exp(-j*phi_reference)"
    return report, data

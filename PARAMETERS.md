# Parameters and Calculation Conventions

This document lists the parameters used by the recorded-data pipeline in
`cornersense/core/pipeline.py`. The [2-dynamic-path README](simulation_2_dynamic_paths/README.md)
and [4-dynamic-path README](simulation_4_dynamic_paths/README.md) describe their
synthetic signal models, the aug-CSI power projection, and figure settings.

## Physical quantities, units, and decisions

| Parameter or quantity | Setting | Role |
|---|---|---|
| Propagation speed c | 299792458 m/s | Conversion between phase and path length |
| Carrier frequency | 5.32 GHz | Displacement readout from carrier phase |
| Effective subcarrier spacing Δf | 625 kHz | Frequency-slope readout and conversion to G |
| Intel 5300 frequencies | 30 nonuniformly selected frequencies | Defined in core/constants.py for the recorded CSI subcarriers |
| tr | abs(Dphys), m | Path metric in metres |
| G | (2πΔf/c)tr, rad | Phase metric used in saved results and plots |
| Recorded-data direction threshold | Ig > 0.10 m or Ig < −0.10 m | Approaching or moving away; Ig is the last-20% median of tr minus its first-20% median |
| No-person output | −0.10 ≤ Ig ≤ +0.10 m | The motion output is `noperson` |
| Recorded-data range threshold T_R | 0.85 m | R1 when min(tr) > T_R; R2 otherwise |

Recorded-data evaluation uses 0.10 m for direction and 0.85 m for range.
R1 and R2 follow the range-region definitions and experimental settings in
the [CornerSense paper](REFERENCES.md).
Candidate screening and calibration have their own parameters.

## Beat-frequency candidate selection

| Setting | Value | Role |
|---|---|---|
| Default temporal band | 0.8–8.0 Hz | Analysis band for recorded data |
| Standard distance-anchor interval | 0.4 ≤ abs(D) ≤ 4.0 m | Accepted interval for candidate anchors |
| Extended distance-anchor interval | 0.2 ≤ abs(D) ≤ 6.0 m and D_iqr ≤ 1.0 m | Wider interval for anchors with low variability |
| PCA phase-lock threshold | 0.5 | A phase-lock score ≥ 0.5 confirms the match between a PCA component and a beat-frequency candidate |
| Consistency weight | 0.5 | Halves the absolute-trend score when at least one cross-antenna/conjugate check is available and all available checks report a mismatch |

Candidates are screened by the distance-anchor criteria. PCA-confirmed
candidates are considered first; if none meet these criteria, all admissible
tracked candidates are considered. Selection maximizes the absolute trend
multiplied by the consistency weight. D is a path-length difference anchor,
and D_iqr is its interquartile range.

## Recorded-data angle calibration

[calibration/sample_angles.json](calibration/sample_angles.json) uses
`method: fixed_session_offset` and an `offsets_deg` mapping keyed by
recording-session folder name.

| Recording folder | Offset b |
|---|---:|
| 0319_data | −3° |
| 0325_data | 22° |
| data_1013 | −12° |
| data_sup01 | 0° |

For each recording, MUSIC estimates the reference-path angle `theta_r` and
human-path angle `theta_u`. Both use the same subtraction:

```text
theta_r_corrected = theta_r_measured − b
theta_u_corrected = theta_u_measured − b
```

Angles and offsets are expressed in degrees. Corrected angles must lie
within [−80°, 80°]; an out-of-range value raises an error. Inference converts
the corrected angles to radians for the robot-motion compensation.

Inference loads one fixed AoA offset per session and applies it to every
recording in that session.

## Related files

- [CornerSense pipeline](cornersense/core/pipeline.py): beat-frequency candidates, PCA, and motion compensation.
- [Calibration settings](calibration/sample_angles.json): fixed per-session AoA offsets.
- [Recorded-data inference](cornersense/inference.py): angle conversion from degrees to radians and pipeline execution.
- [2 dynamic paths](simulation_2_dynamic_paths/README.md): aug-CSI power projection and power comparisons for two dynamic paths with a weak static background.
- [4 dynamic paths](simulation_4_dynamic_paths/README.md): aug-CSI power projection and power comparisons for four dynamic paths.

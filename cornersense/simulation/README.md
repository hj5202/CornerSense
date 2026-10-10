# Shared simulation tools

The [two-dynamic-path](../../simulation_2_dynamic_paths/README.md) and
[four-dynamic-path](../../simulation_4_dynamic_paths/README.md) projects use this
implementation for CSI synthesis, CornerSense processing and comparison figures.
Each project supplies its own configuration and saved results.

## Processing

`generate.py` produces analytic complex CSI and separate path truth.
`pipeline.py` estimates AoA with `measure_angles()` and calls the CornerSense
pipeline's `run_file(preloaded=..., return_intermediates=True)`.
`evaluate.py` exports the returned aug-CSI power projection and applies the same
CornerSense smoothing, bandpass and PCA functions to raw power.

The two branches use the selected CSI segment, first antenna, tone 15 (index 14),
0.2 s smoothing and a 0.8–8 Hz passband. The virtual field is
`Ha = 2 max(abs(CSI)) exp(jπ/4)` on the selected first-antenna segment.

| Curve | Definition |
|---|---|
| Raw power | `abs(H)^2` |
| Augmented power | `abs(H + Ha)^2` |
| aug-CSI power projection | Returned PCA `score * loading[14]` |
| Raw-power projection | Raw-power smoothing, bandpass and PCA; then `score * loading[14]` |
| Cosine truth | `cos(arg(H_reference) − arg(Ha))` |

For the cosine comparison, each projection is centered and divided by its own
`sqrt(2) × RMS`; the normalized component can exceed ±1. Unscaled projections
are also saved. The power comparison uses uncentered power with separate
vertical scales. With `H_reference = A exp(−jφr)` and
`Ha = abs(Ha) exp(jφa)`, the cosine truth is `cos(φr + φa)`.
The truth comparison follows projection extraction.

## Outputs

Each new run writes all of the following files to the selected output directory.
The bundled results contain the files marked below.

| File | Contents | Bundled |
|---|---|---|
| `config.json` | Synthesis settings used for the run | Yes |
| `observation.npz` | Synthetic CSI, times and frequencies | No |
| `truth.npz` | Individual path fields, lengths and synthesis truth | No |
| `stage1.npz` | aug-CSI and raw-power PCA intermediates | No |
| `data.npz` | Plotted curves, timestamps and packet indices | Yes |
| `report.json` | Comparison scores, measured AoA, processing settings and spectral peaks | Yes |
| `figures/` | Cosine and power comparisons, each as PNG and SVG | Yes |

Redrawing the figures requires only `data.npz` and `report.json`.

## Tests

A normal run processes the CSI once. Tests cover pipeline integration, repeated
runs, preprocessing equivalence, truth separation, saved curves and scenario
settings. Run them from the repository root:

```bash
python -B -m unittest discover -s tests -p test_simulation_shared.py -v
python -B -m unittest simulation_2_dynamic_paths.test_simulation simulation_4_dynamic_paths.test_simulation -v
```

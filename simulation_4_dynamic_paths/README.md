# Four-dynamic-path simulation

This simulation generates analytic complex CSI from four dynamic propagation
paths and processes it with CornerSense to produce two figures:

1. **Cosine comparison:** the CornerSense aug-CSI power projection, a raw-power PCA
   control, and the reference–virtual cosine truth.
2. **Power comparison:** raw CSI power and virtual-path augmented CSI power.

## Run

Use the dependencies in the main project's `requirements.txt`. From the
CornerSense root directory:

```bash
python -B -m simulation_4_dynamic_paths.run --output results/simulation_4_dynamic_paths
python -B -m unittest simulation_4_dynamic_paths.test_simulation -v
python -B -m unittest discover -s tests -p test_simulation_shared.py -v
```

Use a new output directory for each run.

The supplied results are in `simulation_4_dynamic_paths/results/`. The command
above saves a new run in `results/simulation_4_dynamic_paths/`. To redraw the
supplied figures:

```bash
python -B -m simulation_4_dynamic_paths.make_figures simulation_4_dynamic_paths/results
```

## Signal settings

The generated input spans 0–8 s at 400 Hz and has 3,201 samples, three antennas
spaced 2.5 cm apart, and the main project's 30-tone frequency grid at 5.32 GHz.
The CornerSense pipeline selects samples 0–3,199 for the plots, ending at 7.9975 s.

| Path | Length at t = 0 | Path-length rate | Amplitude | Array angle | Initial phase |
|---|---:|---:|---:|---:|---:|
| 1. Reference | 20 m | −0.1 m/s | 1 | 30° | 0.5 rad |
| 2. Secondary | 22 m | −0.25 m/s | 0.3 | 60° | 1 rad |
| 3. Third | 24 m | +0.05 m/s | 0.15 | −30° | 1.5 rad |
| 4. Fourth | 26 m | −0.4 m/s | 0.05 | −60° | 2 rad |

Each path has the field
`H_p = A_p R(f) exp[-j 2πf (L_p(t) − d_a sin θ_p)/c + j phase_p]`,
with `L_p(t) = L_p(0) + rate_p t` and a common receiver response
`R(f) = 1/[1 + j(f−fc)/(3.5 MHz)]`. The saved example is noise-free.
The receiver-speed input is 0 m/s. The listed rates are scalar propagation-length
change rates.

The reference path has the same settings as in the two-dynamic-path simulation.
The remaining three paths have progressively weaker amplitudes and distinct
nonzero length-change rates. CornerSense adds a constant virtual field during
processing.

## Processing and displayed quantities

The [shared simulation tools](../cornersense/simulation/README.md) generate CSI,
estimate AoA and process the observations with the CornerSense pipeline.
They export the aug-CSI power projection and apply the same smoothing, bandpass
and PCA functions to raw power. The shared description defines the plotted
quantities, normalization and saved intermediate arrays.

## Results

| Projection vs. cosine truth | Full-window correlation | Display RMSE |
|---|---:|---:|
| aug-CSI power projection | 0.929 | 0.266 |
| Raw-power PCA | 0.009 | 0.994 |

The aug-CSI power projection follows the reference–virtual component, while the
raw projection has a different dominant beat: their largest spectral bins are
1.750 and 2.625 Hz, respectively. The ideal frequencies are 1.775 Hz for the
reference and 2.662 Hz for the secondary–reference difference.

The CornerSense pipeline's separate beat matcher selects −2.625 Hz.
`results/report.json` contains full-window and central-60% comparison scores,
the beat match, dominant spectral bins and processing settings.

## Files

The repository includes:

| File | Contents |
|---|---|
| `config.json` | Synthesis settings |
| `run.py` | Entry point for the shared simulation runner |
| `make_figures.py` | Entry point for the shared figure renderer |
| `test_simulation.py` | Four-path scenario settings |
| `results/config.json` | Synthesis settings used for the saved results |
| `results/data.npz` | Plotted curves, timestamps and packet indices |
| `results/report.json` | Comparison scores, measured AoA and processing settings |
| `results/figures/` | `cosine_comparison` and `power_comparison`, each as PNG and SVG |

Each new run saves these results and the following additional files in the
selected output directory:

| File | Contents |
|---|---|
| `observation.npz` | Synthetic input CSI, times and frequencies |
| `truth.npz` | Individual path fields and lengths |
| `stage1.npz` | aug-CSI power projection arrays and raw-control arrays |

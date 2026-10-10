# Data and permitted use

The project owner makes the real-data samples and simulation data in this
repository available for non-commercial use under CC BY-NC 4.0. See LICENSE
for the license notice and full legal-terms link. Cite the CornerSense paper,
credit this project, provide the license link, and identify modifications
when sharing the data or derived figures.

## Real recordings

sample/ contains 35 selected Intel 5300 CSI recordings: 0319_data (5),
0325_data (5), data_1013 (9), and data_sup01 (16). The filenames use a common
scene, motion, range, and repeat convention; the original CSI bytes are
retained. Each dated folder represents one recording session and power cycle,
with one to three scene variants. The set contains 34 human-motion recordings
and one no-person recording, `csi_0319_class_noperson_all_m1`. See
[sample/README.md](sample/README.md) for the naming scheme.

The [project README](README.md#saved-results) describes the saved results and
their calibration basis.

## Synthetic recordings

The repository includes two analytic CSI experiments:

- [2 dynamic paths](simulation_2_dynamic_paths/README.md): two dynamic paths
  and one weak static background path.
- [4 dynamic paths](simulation_4_dynamic_paths/README.md): four dynamic paths
  with decreasing amplitudes and different path-length change rates.

Both use 8 s of noiseless three-antenna, 30-subcarrier CSI sampled at 400 Hz,
with a stationary receiver.

Each project includes plotted curves, scores, a configuration snapshot, and
two figures: normalized components compared with cosine truth, and raw versus
augmented power. A new run also generates synthetic CSI, analytic path truth,
and pipeline intermediate arrays. The project READMEs describe the signal
models, settings, and reproduction commands.

## Attribution and third-party material

The non-commercial grant covers material owned by the project authors. It
does not relicense NumPy, SciPy, Matplotlib, or third-party code. See
REFERENCES.md for the paper and AoA method attribution.

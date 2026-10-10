# Recorded Samples

This directory contains 35 Intel 5300 CSI recordings in four dated folders.
Each folder represents one recording session and power cycle, with one to
three scene variants. Recordings use extensionless filenames.

| Folder | Recordings | Contents |
|---|---:|---|
| 0319_data | 5 | Four human-motion recordings and one no-person recording |
| 0325_data | 5 | Human-motion recordings |
| data_1013 | 9 | Human-motion recordings |
| data_sup01 | 16 | Human-motion recordings |

## Filename fields

- `approaching` and `movingaway`: human-motion labels.
- `noperson`: no person or robot-only motion; range evaluation is not applicable.
- `class`, `corridor`, and `stage`: classroom, corridor, and stage setups.
- `r1`, `r2`, and `all`: the recorded range protocol. The `all` label uses
  separate range evaluation for the first and second halves of the recording.
  R1 and R2 follow the range-region definitions and settings in the paper.
- `m1`, `m2`, ...: repeat numbers assigned in the original file-size order.

The bundled recordings are processed at a robot speed of 0.2 m/s.

## Run

From the repository root, process all four folders with:

```bash
python run.py --input sample
```

See the [project README](../README.md) for calibration, metrics, and saved results.

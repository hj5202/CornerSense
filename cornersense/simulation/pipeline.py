"""Estimate AoA and process synthetic observations with CornerSense."""
from contextlib import redirect_stdout
import io
import sys

import numpy as np

from .. import inference
from ..core import pipeline


def detect(csi, times, robot_speed):
    """Run CornerSense on observed CSI, timestamps and receiver speed."""
    diagnostics = io.StringIO()
    try:
        with redirect_stdout(diagnostics):
            measured = inference.measure_angles(csi)
            motion = {"th_r": float(np.deg2rad(measured["theta_r"])),
                      "th_u": float(np.deg2rad(measured["theta_u"])),
                      "robot_speed": float(robot_speed)}
            result = pipeline.run_file("synthetic_input.npz", motion,
                preloaded=(csi, times), return_intermediates=True)
    except Exception:
        print(diagnostics.getvalue(), end="", file=sys.stderr)
        raise
    for line in diagnostics.getvalue().splitlines():
        if "WARNING" in line.upper():
            print(line, file=sys.stderr)
    return {"result": result, "measured_aoa_deg": measured, "motion_geometry": motion}

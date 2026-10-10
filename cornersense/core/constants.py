"""Radio propagation constants and the Intel 5300 subcarrier grid.

SUB lists the recorded tones; FREQS gives their absolute frequencies in Hz.
"""
import numpy as np

C = 299_792_458.0
FC = 5.32e9
DF = 312_500.0

SUB = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10,
                -8, -6, -4, -2, -1, 1, 3, 5, 7, 9,
                11, 13, 15, 17, 19, 21, 23, 25, 27, 28], dtype=float)
FREQS = FC + SUB * DF

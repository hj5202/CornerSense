"""Intel 5300 CSI binary decoding and channel scaling.

read_bf_file returns a CSITrace; remove_spatial_mapping restores transmit channels.
The Linux 802.11n CSI Tool format reference is listed in REFERENCES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass
class CSITrace:
    csi: np.ndarray
    timestamps_s: np.ndarray
    bfee_count: np.ndarray
    rssi: np.ndarray
    noise: np.ndarray
    agc: np.ndarray
    rate_flags: np.ndarray
    dropped_records: int
    source: Path

    @property
    def num_packets(self) -> int:
        return int(self.csi.shape[0])

    @property
    def num_tx(self) -> int:
        return int(self.csi.shape[1])

    @property
    def num_rx(self) -> int:
        return int(self.csi.shape[2])

    @property
    def num_subcarriers(self) -> int:
        return int(self.csi.shape[3])

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1]) if self.num_packets > 1 else 0.0

    @property
    def sample_rate_hz(self) -> float:
        if self.duration_s <= 0.0 or self.num_packets < 2:
            return 0.0
        return float((self.num_packets - 1) / self.duration_s)


def _dbinv(value: float) -> float:
    return 10.0 ** (value / 10.0)


def _total_rss_dbm(rssi: np.ndarray, agc: int) -> float:
    magnitude = sum(_dbinv(float(v)) for v in rssi if v != 0)
    if magnitude <= 0.0:
        return -120.0
    return 10.0 * np.log10(magnitude) - 44.0 - float(agc)


def _decode_csi(payload: bytes, nrx: int, ntx: int) -> np.ndarray:
    expected = (30 * (nrx * ntx * 8 * 2 + 3) + 7) // 8
    declared = payload[16] | (payload[17] << 8)
    if declared != expected or len(payload) < 20 + expected:
        raise ValueError("invalid beamforming matrix length")
    packed = payload[20:20 + expected]
    csi = np.empty((ntx, nrx, 30), dtype=np.complex128)
    bit_index = 0
    for subcarrier in range(30):
        bit_index += 3
        remainder = bit_index & 7
        byte_index = bit_index >> 3
        for linear_index in range(nrx * ntx):
            real_u8 = ((packed[byte_index] >> remainder) |
                       (packed[byte_index + 1] << (8 - remainder))) & 0xFF
            imag_u8 = ((packed[byte_index + 1] >> remainder) |
                       (packed[byte_index + 2] << (8 - remainder))) & 0xFF
            real = real_u8 - 256 if real_u8 >= 128 else real_u8
            imag = imag_u8 - 256 if imag_u8 >= 128 else imag_u8
            tx = linear_index % ntx
            rx = linear_index // ntx
            csi[tx, rx, subcarrier] = complex(real, imag)
            bit_index += 16
            byte_index = bit_index >> 3
    return csi


def _scale_csi(csi: np.ndarray, rssi: np.ndarray, noise: int, agc: int) -> np.ndarray:
    csi_power = float(np.sum(np.abs(csi) ** 2))
    if csi_power <= 0.0:
        return csi.copy()
    rssi_power = _dbinv(_total_rss_dbm(rssi, agc))
    scale = rssi_power / (csi_power / 30.0)
    noise_db = -92 if noise == -127 else int(noise)
    thermal_noise = _dbinv(float(noise_db))
    quantization_error = scale * csi.shape[0] * csi.shape[1]
    scaled = csi * np.sqrt(scale / max(thermal_noise + quantization_error,
                                      np.finfo(float).tiny))
    if csi.shape[0] == 2:
        scaled *= np.sqrt(2.0)
    elif csi.shape[0] == 3:
        scaled *= np.sqrt(_dbinv(4.5))
    return scaled


def _unwrap_timestamps(raw: np.ndarray) -> np.ndarray:
    raw64 = raw.astype(np.int64)
    increments = np.diff(raw64)
    wraps = increments < -(2**31)
    wrap_count = np.concatenate(([0], np.cumsum(wraps, dtype=np.int64)))
    unwrapped = raw64 + wrap_count * (2**32)
    return (unwrapped - unwrapped[0]).astype(float) / 1e6


def spatial_mapping_matrix(ntx: int, rate: int) -> np.ndarray:
    is_40mhz = bool(rate & 2048)
    if ntx == 1:
        return np.ones((1, 1), dtype=np.complex128)
    if ntx == 2:
        values = (np.array([[1, 1j], [1j, 1]], dtype=np.complex128)
                  if is_40mhz else
                  np.array([[1, 1], [1, -1]], dtype=np.complex128))
        return values / np.sqrt(2.0)
    if ntx == 3 and not is_40mhz:
        phase = np.array([
            [-2*np.pi/16, -2*np.pi/(80/33), 2*np.pi/(80/3)],
            [2*np.pi/(80/23), 2*np.pi/(48/13), 2*np.pi/(240/13)],
            [-2*np.pi/(80/13), 2*np.pi/(240/37), 2*np.pi/(48/13)],
        ])
    elif ntx == 3:
        phase = np.array([
            [-2*np.pi/16, -2*np.pi/(80/13), 2*np.pi/(80/23)],
            [-2*np.pi/(80/37), -2*np.pi/(48/11), -2*np.pi/(240/107)],
            [2*np.pi/(80/7), -2*np.pi/(240/83), -2*np.pi/(48/11)],
        ])
    else:
        raise ValueError(f"unsupported Ntx={ntx}")
    return np.exp(1j * phase) / np.sqrt(3.0)


def remove_spatial_mapping(csi: np.ndarray, rate: int) -> np.ndarray:
    matrix = spatial_mapping_matrix(csi.shape[0], rate)
    return np.einsum("ab,bck->ack", np.conj(matrix), csi)


def read_bf_file(path, scale: bool = True,
                 remove_sm: bool = False) -> CSITrace:
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"CSI trace is missing or empty: {source}")
    records = []
    dropped = 0
    with source.open("rb") as handle:
        while True:
            header = handle.read(3)
            if not header:
                break
            if len(header) != 3:
                dropped += 1
                break
            field_len = struct.unpack(">H", header[:2])[0]
            code = header[2]
            body = handle.read(max(0, field_len - 1))
            if len(body) != field_len - 1:
                dropped += 1
                break
            if code != 187 or len(body) < 20:
                continue
            nrx, ntx = body[8], body[9]
            if nrx not in (1, 2, 3) or ntx not in (1, 2, 3):
                dropped += 1
                continue
            try:
                csi = _decode_csi(body, nrx, ntx)
            except (ValueError, IndexError):
                dropped += 1
                continue
            antenna_sel = body[15]
            permutation = [antenna_sel & 0x3,
                           (antenna_sel >> 2) & 0x3,
                           (antenna_sel >> 4) & 0x3]
            if nrx > 1 and sorted(permutation[:nrx]) == list(range(nrx)):
                reordered = np.empty_like(csi)
                for raw_rx, physical_rx in enumerate(permutation[:nrx]):
                    reordered[:, physical_rx, :] = csi[:, raw_rx, :]
                csi = reordered
            timestamp = struct.unpack("<I", body[0:4])[0]
            count = struct.unpack("<H", body[4:6])[0]
            rssi = np.array([body[10], body[11], body[12]], dtype=np.int16)
            noise = struct.unpack("b", body[13:14])[0]
            agc = body[14]
            rate = struct.unpack("<H", body[18:20])[0]
            if scale:
                csi = _scale_csi(csi, rssi, noise, agc)
            if remove_sm and ntx > 1:
                csi = remove_spatial_mapping(csi, rate)
            records.append((timestamp, count, rssi, noise, agc, rate, csi))
    if not records:
        raise ValueError(f"no valid CSI records in {source}")
    shapes = {item[-1].shape for item in records}
    if len(shapes) != 1:
        dominant = max(shapes,
                       key=lambda shape: sum(x[-1].shape == shape for x in records))
        kept = [x for x in records if x[-1].shape == dominant]
        dropped += len(records) - len(kept)
        records = kept
    raw_timestamps = np.array([x[0] for x in records], dtype=np.uint32)
    return CSITrace(
        csi=np.stack([x[-1] for x in records]),
        timestamps_s=_unwrap_timestamps(raw_timestamps),
        bfee_count=np.array([x[1] for x in records], dtype=np.uint16),
        rssi=np.stack([x[2] for x in records]),
        noise=np.array([x[3] for x in records], dtype=np.int16),
        agc=np.array([x[4] for x in records], dtype=np.int16),
        rate_flags=np.array([x[5] for x in records], dtype=np.uint16),
        dropped_records=dropped,
        source=source,
    )

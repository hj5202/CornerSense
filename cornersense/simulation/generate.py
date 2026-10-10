"""Generate three-antenna CSI from explicitly defined scalar propagation paths."""
import numpy as np


def generate(config, frequencies, light_speed):
    fs = config["sample_rate_hz"]
    count = int(round(config["duration_s"] * fs)) + 1
    times = np.arange(count) / fs
    frequencies = np.asarray(frequencies, dtype=float)
    offsets = np.asarray(config["antenna_offsets_m"], dtype=float)
    center = config["carrier_hz"]
    bandwidth = config["receiver_bandwidth_hz"]
    response = 1 / (1 + 1j * (frequencies - center) / bandwidth)
    fields, lengths, rates = [], [], []
    for path in config["paths"]:
        length = path["length_m"] + path["rate_mps"] * times
        spatial = offsets * np.sin(np.deg2rad(path["angle_deg"]))
        # Positive spatial phase follows the SpotFi steering convention.
        phase = -2 * np.pi * (length[:, None, None] - spatial[None, :, None])
        phase = phase * frequencies[None, None, :] / light_speed + path["phase_rad"]
        fields.append(path["amplitude"] * np.exp(1j * phase) * response[None, None, :])
        lengths.append(length)
        rates.append(path["rate_mps"])
    fields = np.asarray(fields)
    rng = np.random.default_rng(config["seed"])
    noise = config["noise_std"] * (rng.standard_normal(fields.shape[1:]) +
                                  1j * rng.standard_normal(fields.shape[1:]))
    observation = {"csi": fields.sum(axis=0) + noise, "time_s": times,
                   "frequencies_hz": frequencies,
                   "robot_speed_mps": np.asarray(config["robot_speed_mps"])}
    truth = {"time_s": times, "path_fields": fields, "path_lengths_m": np.asarray(lengths),
             "path_names": np.asarray([p["name"] for p in config["paths"]]),
             "path_rates_mps": np.asarray(rates), "noise": noise,
             "receiver_response": response,
             "path_doppler_hz": -center * np.asarray(rates) / light_speed}
    return observation, truth

# References and attribution

## CornerSense

Jing He, Rui Zhang, Qijia Wang, Ruiqi Kong, and He Chen. 2025.
**CornerSense: Virtual Path-augmented WiFi Sensing for Human Proximity
Detection Around Corners on Mobile Robots.** Proceedings of the ACM on
Interactive, Mobile, Wearable and Ubiquitous Technologies 9(4), Article 181,
December 2025. DOI: https://doi.org/10.1145/3770699.

ACM 2474-9567/2025/12-ART181. Copyright (c) 2025 held by the owner/author(s).

## AoA methods

The AoA stage follows **SpotFi**. cornersense/core/spotfi.py provides CSI phase
preprocessing, spatial/frequency-smoothed MUSIC, and angle-delay peak selection.

Manikanta Kotaru, Kiran Joshi, Dinesh Bharadia, and Sachin Katti. 2015.
**SpotFi: Decimeter Level Localization Using WiFi.** Proceedings of ACM
SIGCOMM 2015.

## Intel 5300 recording format

The recording format decoded by cornersense/core/intel5300.py is documented
by the **Linux 802.11n CSI Tool**. It reports 30 subcarrier groups, with
signed 8-bit real and imaginary components for each antenna-pair entry.

Daniel Halperin, Wenjun Hu, Anmol Sheth, and David Wetherall. 2011.
**Tool release: Gathering 802.11n traces with channel state information.**
ACM SIGCOMM Computer Communication Review, January 2011.

- [CSI Tool documentation](https://dhalperi.github.io/linux-80211n-csitool/)
- [Upstream recording and parsing tools](https://github.com/dhalperi/linux-80211n-csitool-supplementary)

## Dependencies

NumPy, SciPy, and Matplotlib are installed separately and retain their own
licenses. This repository's CC BY-NC notice applies to the project authors'
material.

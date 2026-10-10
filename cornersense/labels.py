"""Parse evaluation labels from the published sample filename convention.

label_for returns the session, motion, range and repetition for a recording.
"""
from __future__ import annotations

import re
from pathlib import Path


def label_for(path) -> dict:
    name = Path(path).name
    lab = {"file": name, "session": None, "motion": "unknown", "region": None,
           "region_origin": None, "repeat": None}
    match = re.fullmatch(
        r"(csi_0319_class|csi_0325_class|csi0103_(?:class|stage)|1013_(?:corridor|stage))"
        r"_(approaching|movingaway|noperson)_(all|r1|r2)_m(\d+)", name)
    if match is None:
        return lab
    prefix, motion, region, repeat = match.groups()
    sessions = {"csi_0319_class": "0319", "csi_0325_class": "0325",
                "csi0103_class": "sup01_class", "csi0103_stage": "sup01_stage",
                "1013_corridor": "1013_corridor", "1013_stage": "1013_stage"}
    lab.update(session=sessions[prefix], repeat=int(repeat),
               motion={"movingaway": "moving_away", "noperson": "no_person"}.get(motion, motion))
    if lab["motion"] != "no_person":
        lab["region"] = {"r1": "R1", "r2": "R2", "all": "temporal_half"}[region]
        lab["region_origin"] = "temporal_half" if region == "all" else "explicit"
    return lab

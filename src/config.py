"""Load the YAML configuration files."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_lab(path: str | Path) -> dict:
    """Load bottles.yaml and index it by tag ID."""
    lab = load_yaml(path)
    lab.setdefault("bottles", [])
    lab.setdefault("containers", [])
    lab.setdefault("incompatible", [])

    tag_to_bottle_body: dict[int, str] = {}
    tag_to_bottle_cap: dict[int, str] = {}
    tag_to_container: dict[int, str] = {}
    seen: dict[int, str] = {}

    def claim(tag: int, owner: str) -> None:
        if tag in seen:
            raise ValueError(f"{path}: tag {tag} used by both {seen[tag]} and {owner}")
        seen[tag] = owner

    for b in lab["bottles"]:
        claim(b["body_tag"], f"{b['name']} body")
        tag_to_bottle_body[b["body_tag"]] = b["name"]
        if b.get("cap_tag") is not None:
            claim(b["cap_tag"], f"{b['name']} cap")
            tag_to_bottle_cap[b["cap_tag"]] = b["name"]
    for c in lab["containers"]:
        claim(c["tag"], c["name"])
        tag_to_container[c["tag"]] = c["name"]
    if lab.get("table_tag") is not None:
        claim(lab["table_tag"], "table")

    lab["tag_to_bottle_body"] = tag_to_bottle_body
    lab["tag_to_bottle_cap"] = tag_to_bottle_cap
    lab["tag_to_container"] = tag_to_container

    # Physical size of every tag (metres): caps use smaller tags.
    default = float(lab.get("tag_size_m", 0.030))
    cap = float(lab.get("cap_tag_size_m", default))
    lab["tag_sizes"] = {tag: (cap if tag in tag_to_bottle_cap else default) for tag in seen}
    return lab

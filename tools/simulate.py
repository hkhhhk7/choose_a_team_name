"""Replay a scripted list of bench events through the SOP checker (no camera).

    python tools/simulate.py samples/scenario_wrong_order.yaml

Scenario file:
    sop: configs/sop/dilute_acid.yaml
    lab: configs/bottles.yaml
    events:
      - [1.0, enter, beaker_A]         # a bottle or a container comes into view
      - [1.5, enter, acid_conc]
      - [2.5, pour, acid_conc, beaker_A]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_lab, load_yaml  # noqa: E402
from src.records import Event  # noqa: E402
from src.sop_checker import SOPChecker  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sc = load_yaml(sys.argv[1])
    lab = load_lab(sc["lab"])
    containers = {c["name"] for c in lab["containers"]}
    checker = SOPChecker(load_yaml(sc["sop"]), lab)
    for row in sc["events"]:
        t, kind, name, *rest = row
        if name in containers:
            ev = Event(float(t), kind, None, name)
        else:
            ev = Event(float(t), kind, name, rest[0] if rest else None)
        print(f"  {t:6.1f}s  {kind:9s} {name}{' -> ' + rest[0] if rest else ''}")
        for a in checker.process(ev):
            print("        ", a)
    if "end" in sc:
        for a in checker.tick(float(sc["end"])):
            print("        ", a)
    print(json.dumps(checker.summary(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

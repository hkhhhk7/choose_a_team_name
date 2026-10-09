"""SOP checker: compares bench events with the expected procedure.

The checker is a small state machine. Its state is
  * which steps are done (and when),
  * what each container holds,
  * which caps are open (and since when).
Every incoming Event updates that state and may produce Alerts.

Two kinds of alerts exist:
  * warnings (orange), raised when a bottle or a container ENTERS the work
    area and does not fit the next step, before anything is poured;
  * errors (red, level DANGER), raised on POUR, when the student went ahead
    anyway or broke a safety rule.
The warnings are what lets LabGuardian warn *before* the mistake.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from .records import (
    CAP_CLOSE,
    CAP_OPEN,
    DANGER,
    ENTER,
    INFO,
    LEAVE,
    POUR,
    WARNING,
    Alert,
    Event,
)


@dataclass
class Step:
    id: str
    action: str
    reagent: str
    into: str
    after: tuple[str, ...] = ()


@dataclass
class Forbidden:
    action: str
    reagent: str
    message: str
    into: Optional[str] = None  # a specific container
    into_containing: Optional[str] = None  # any container already holding this


def _as_tuple(x) -> tuple[str, ...]:
    if x is None:
        return ()
    if isinstance(x, str):
        return (x,)
    return tuple(x)


class SOPChecker:
    def __init__(self, sop: dict, lab: Optional[dict] = None):
        self.name: str = sop.get("name", "unnamed procedure")
        self.steps: list[Step] = [
            Step(
                id=s["id"],
                action=s.get("action", "pour"),
                reagent=s["reagent"],
                into=s["into"],
                after=_as_tuple(s.get("after")),
            )
            for s in sop.get("steps", [])
        ]
        self.forbidden: list[Forbidden] = [
            Forbidden(
                action=f.get("action", "pour"),
                reagent=f["reagent"],
                message=f["message"],
                into=f.get("into"),
                into_containing=f.get("into_containing"),
            )
            for f in sop.get("forbidden", [])
        ]
        self.cap_open_timeout_s: Optional[float] = sop.get("cap_open_timeout_s")

        step_ids = {s.id for s in self.steps}
        if len(step_ids) != len(self.steps):
            raise ValueError("duplicate step id in SOP")
        for s in self.steps:
            for p in s.after:
                if p not in step_ids:
                    raise ValueError(f"step {s.id!r} depends on unknown step {p!r}")

        lab = lab or {}
        self.labels: dict[str, str] = {}
        for item in lab.get("bottles", []) + lab.get("containers", []):
            self.labels[item["name"]] = item.get("label", item["name"])
        self.incompatible: set[frozenset[str]] = {
            frozenset(pair) for pair in lab.get("incompatible", [])
        }

        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self.done: dict[str, float] = {}
        self.contents: dict[str, list[str]] = defaultdict(list)
        self.open_caps: dict[str, float] = {}
        self._cap_warned: set[str] = set()
        self.completed_at: Optional[float] = None
        self.bottles_in_view: set[str] = set()
        self.containers_in_view: set[str] = set()
        self.events: list[Event] = []
        self.alerts: list[Alert] = []

    @property
    def complete(self) -> bool:
        return self.completed_at is not None

    @property
    def problems(self) -> list[Alert]:
        """Warnings and dangers, excluding the end-of-procedure summary."""
        return [a for a in self.alerts
                if a.level in (WARNING, DANGER) and not a.code.startswith("COMPLETE")]

    def _label(self, name: Optional[str]) -> str:
        return self.labels.get(name, name) if name else "?"

    def _step_words(self, ids) -> str:
        """'add_water' -> 'Distilled water into Beaker A', for messages."""
        by_id = {s.id: s for s in self.steps}
        return ", ".join(f"{self._label(by_id[i].reagent)} into {self._label(by_id[i].into)}" for i in ids)

    def _pending(self) -> list[Step]:
        return [s for s in self.steps if s.id not in self.done]

    def _ready(self, step: Step) -> bool:
        return all(p in self.done for p in step.after)

    def next_steps(self) -> list[Step]:
        """Steps the student may do right now."""
        return [s for s in self._pending() if self._ready(s)]

    # ----------------------------------------------------------------- events

    def process(self, ev: Event) -> list[Alert]:
        self.events.append(ev)
        if ev.kind == ENTER and ev.bottle:
            self.bottles_in_view.add(ev.bottle)
            out = self._on_bottle_enter(ev)
        elif ev.kind == ENTER and ev.container:
            self.containers_in_view.add(ev.container)
            out = self._on_container_enter(ev)
        elif ev.kind == LEAVE:
            self.bottles_in_view.discard(ev.bottle)
            self.containers_in_view.discard(ev.container)
            out = []
        elif ev.kind == POUR:
            out = self._on_pour(ev)
        elif ev.kind == CAP_OPEN:
            self.open_caps.setdefault(ev.bottle, ev.t)
            out = []
        elif ev.kind == CAP_CLOSE:
            self.open_caps.pop(ev.bottle, None)
            self._cap_warned.discard(ev.bottle)
            out = []
        else:  # pragma: no cover - Event already validates kind
            out = []
        out += self.tick(ev.t)
        self.alerts += out
        return out

    def process_all(self, events: Iterable[Event]) -> list[Alert]:
        out: list[Alert] = []
        for ev in events:
            out += self.process(ev)
        return out

    def tick(self, t: float) -> list[Alert]:
        """Time-based checks; call this every frame even without events."""
        out: list[Alert] = []
        if self.cap_open_timeout_s is not None:
            for bottle, since in self.open_caps.items():
                if bottle in self._cap_warned:
                    continue
                if t - since >= self.cap_open_timeout_s:
                    self._cap_warned.add(bottle)
                    out.append(
                        Alert(
                            t, WARNING, "CAP_OPEN",
                            f"{self._label(bottle)} has been open for "
                            f"{t - since:.0f} s. Close the cap.",
                            bottle=bottle,
                        )
                    )
        return out

    # -- enter: early warnings ---------------------------------------------

    def _on_container_enter(self, ev: Event) -> list[Alert]:
        c = ev.container
        targets = [s for s in self.steps if s.into == c]
        if not targets:
            return [Alert(ev.t, WARNING, "WRONG_CONTAINER",
                          f"{self._label(c)} is not used in '{self.name}'. Put it back.", container=c)]
        pending = [s for s in targets if s.id not in self.done]
        if not pending:
            return [Alert(ev.t, WARNING, "NOT_NEEDED",
                          f"{self._label(c)} is not needed any more.", container=c)]
        if not any(self._ready(s) for s in pending):
            nxt = self.next_steps()
            hint = f" Next: {self._label(nxt[0].reagent)} into {self._label(nxt[0].into)}." if nxt else ""
            return [Alert(ev.t, WARNING, "OUT_OF_ORDER",
                          f"Not yet: {self._label(c)} is for a later step.{hint}",
                          container=c, step=pending[0].id)]
        return []

    def _on_bottle_enter(self, ev: Event) -> list[Alert]:
        b = ev.bottle
        out: list[Alert] = []

        # Danger if pouring this bottle anywhere it could go would be forbidden.
        for rule in self.forbidden:
            if rule.action != "pour" or rule.reagent != b:
                continue
            for container, held in self.contents.items():
                if self._rule_hits(rule, container, held):
                    out.append(
                        Alert(
                            ev.t, WARNING, "CAUTION",
                            f"Careful: {self._label(container)} holds "
                            f"{self._label(rule.into_containing or held[-1])}. {rule.message}",
                            bottle=b, container=container,
                        )
                    )
        for container, held in self.contents.items():
            for other in held:
                if frozenset((b, other)) in self.incompatible:
                    out.append(
                        Alert(
                            ev.t, WARNING, "CAUTION",
                            f"Do not mix {self._label(b)} with {self._label(other)} "
                            f"(already in {self._label(container)}).",
                            bottle=b, container=container,
                        )
                    )

        uses = [s for s in self.steps if s.reagent == b]
        if not uses:
            out.append(
                Alert(
                    ev.t, WARNING, "WRONG_REAGENT",
                    f"{self._label(b)} is not used in '{self.name}'. Put it back.",
                    bottle=b,
                )
            )
            return out

        pending = [s for s in uses if s.id not in self.done]
        if not pending:
            out.append(
                Alert(
                    ev.t, WARNING, "ALREADY_ADDED",
                    f"{self._label(b)} has already been added.",
                    bottle=b,
                )
            )
            return out

        if not any(self._ready(s) for s in pending):
            missing = sorted({p for s in pending for p in s.after if p not in self.done})
            out.append(
                Alert(
                    ev.t, WARNING, "OUT_OF_ORDER",
                    f"Not yet: first add {self._step_words(missing)}.",
                    bottle=b, step=pending[0].id,
                )
            )
        return out

    # -- pour: violations ---------------------------------------------------

    def _rule_hits(self, rule: Forbidden, container: str, held: list[str]) -> bool:
        if rule.into is not None and rule.into != container:
            return False
        if rule.into_containing is not None and rule.into_containing not in held:
            return False
        return rule.into is not None or rule.into_containing is not None

    def _on_pour(self, ev: Event) -> list[Alert]:
        b, c = ev.bottle, ev.container
        held = self.contents[c]
        out: list[Alert] = []

        for rule in self.forbidden:
            if rule.action == "pour" and rule.reagent == b and self._rule_hits(rule, c, held):
                out.append(Alert(ev.t, DANGER, "FORBIDDEN", rule.message, bottle=b, container=c))
        for other in held:
            if frozenset((b, other)) in self.incompatible:
                out.append(
                    Alert(
                        ev.t, DANGER, "INCOMPATIBLE",
                        f"{self._label(b)} poured into {self._label(c)}, "
                        f"which holds {self._label(other)}.",
                        bottle=b, container=c,
                    )
                )

        exact = [s for s in self._pending() if s.action == "pour" and s.reagent == b and s.into == c]
        if exact:
            # Prefer a step whose prerequisites are met.
            step = next((s for s in exact if self._ready(s)), exact[0])
            missing = [p for p in step.after if p not in self.done]
            if missing:
                out.append(
                    Alert(
                        ev.t, DANGER, "OUT_OF_ORDER",
                        f"Error: {self._label(b)} was added before {self._step_words(missing)}.",
                        bottle=b, container=c, step=step.id,
                    )
                )
            self.done[step.id] = ev.t
        else:
            same_reagent = [s for s in self._pending() if s.reagent == b]
            if same_reagent:
                out.append(
                    Alert(
                        ev.t, DANGER, "WRONG_CONTAINER",
                        f"Error: {self._label(b)} should go into "
                        f"{self._label(same_reagent[0].into)}, not {self._label(c)}.",
                        bottle=b, container=c, step=same_reagent[0].id,
                    )
                )
            elif any(s.reagent == b for s in self.steps):
                out.append(
                    Alert(
                        ev.t, DANGER, "EXTRA_ADDITION",
                        f"Error: {self._label(b)} was added again to {self._label(c)}.",
                        bottle=b, container=c,
                    )
                )
            else:
                out.append(
                    Alert(
                        ev.t, DANGER, "WRONG_REAGENT",
                        f"Error: {self._label(b)} is not part of '{self.name}'.",
                        bottle=b, container=c,
                    )
                )

        held.append(b)

        if not self.complete and not self._pending():
            self.completed_at = ev.t
            problems = self.problems + [a for a in out if a.level in (WARNING, DANGER)]
            if problems:
                n_d = sum(a.level == DANGER for a in problems)
                n_w = len(problems) - n_d
                msg = f"All steps done, with {n_d} error(s) and {n_w} warning(s). Tell your TA."
                out.append(Alert(ev.t, WARNING, "COMPLETE_WITH_ISSUES", msg))
            else:
                out.append(Alert(ev.t, INFO, "COMPLETE", f"'{self.name}' finished correctly."))
        return out

    # ----------------------------------------------------------------- report

    def summary(self) -> dict:
        """Structured summary for the post-lab report."""
        by_level = {lvl: 0 for lvl in (INFO, WARNING, DANGER)}
        for a in self.problems + [a for a in self.alerts if a.level == INFO]:
            by_level[a.level] += 1
        order = sorted(self.done.items(), key=lambda kv: kv[1])
        return {
            "procedure": self.name,
            "completed": self.complete,
            "completed_at_s": self.completed_at,
            "steps_done": [{"step": sid, "t": t} for sid, t in order],
            "steps_missing": [s.id for s in self._pending()],
            "alerts": by_level,
            "alert_log": [str(a) for a in self.alerts],
            "contents": {c: list(v) for c, v in self.contents.items() if v},
            "errors": [str(a) for a in self.problems if a.level == DANGER],
            "warnings": [str(a) for a in self.problems if a.level == WARNING],
        }

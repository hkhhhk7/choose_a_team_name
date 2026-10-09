import unittest

from src.records import CAP_CLOSE, CAP_OPEN, DANGER, ENTER, INFO, LEAVE, POUR, WARNING, Event
from src.sop_checker import SOPChecker

SOP = {
    "name": "Dilute concentrated acid",
    "cap_open_timeout_s": 30,
    "steps": [
        {"id": "add_water", "action": "pour", "reagent": "water", "into": "beaker_A"},
        {"id": "add_acid", "action": "pour", "reagent": "acid_conc", "into": "beaker_A",
         "after": "add_water"},
    ],
    "forbidden": [
        {"action": "pour", "reagent": "water", "into_containing": "acid_conc",
         "message": "Never add water to concentrated acid. Add acid to water."},
    ],
}

LAB = {
    "bottles": [
        {"name": "water", "body_tag": 1},
        {"name": "acid_conc", "body_tag": 3},
        {"name": "bleach", "body_tag": 5},
    ],
    "containers": [{"name": "beaker_A", "tag": 10}, {"name": "beaker_B", "tag": 11}],
    "incompatible": [["bleach", "acid_conc"]],
}


def enter(t, name):
    return Event(t, ENTER, container=name) if name.startswith("beaker") else Event(t, ENTER, name)


def codes(alerts):
    return [a.code for a in alerts]


class SOPCheckerTest(unittest.TestCase):
    def setUp(self):
        self.c = SOPChecker(SOP, LAB)

    def test_correct_procedure_has_no_warnings(self):
        alerts = self.c.process_all([
            enter(0, "beaker_A"),
            enter(1, "water"),
            Event(2, POUR, "water", "beaker_A"),
            Event(3, LEAVE, "water"),
            enter(4, "acid_conc"),
            Event(5, POUR, "acid_conc", "beaker_A"),
            Event(6, LEAVE, "acid_conc"),
        ])
        self.assertEqual(codes(alerts), ["COMPLETE"])
        self.assertEqual(alerts[0].level, INFO)
        self.assertTrue(self.c.complete)
        self.assertEqual(self.c.summary()["steps_missing"], [])
        self.assertEqual(self.c.bottles_in_view, set())

    def test_acid_first_warns_when_it_enters(self):
        alerts = self.c.process(enter(1, "acid_conc"))
        self.assertEqual(codes(alerts), ["OUT_OF_ORDER"])
        self.assertEqual(alerts[0].level, WARNING)

    def test_warned_and_poured_anyway_is_an_error(self):
        self.c.process(enter(1, "acid_conc"))
        pour_acid = self.c.process(Event(2, POUR, "acid_conc", "beaker_A"))
        self.assertEqual(codes(pour_acid), ["OUT_OF_ORDER"])
        self.assertEqual(pour_acid[0].level, DANGER)

        self.assertEqual(codes(self.c.process(enter(3, "water"))), ["CAUTION"])
        pour_water = self.c.process(Event(4, POUR, "water", "beaker_A"))
        self.assertIn("FORBIDDEN", codes(pour_water))
        self.assertIn("COMPLETE_WITH_ISSUES", codes(pour_water))  # both steps happened, wrongly

    def test_wrong_beaker_warns_on_entry_then_errors_on_pour(self):
        self.c.process_all([enter(0, "beaker_A"), enter(1, "water"),
                            Event(2, POUR, "water", "beaker_A")])
        self.assertEqual(codes(self.c.process(enter(3, "acid_conc"))), [])  # acid is next: fine
        b = self.c.process(enter(4, "beaker_B"))
        self.assertEqual(codes(b), ["WRONG_CONTAINER"])
        self.assertEqual(b[0].level, WARNING)
        pour = self.c.process(Event(5, POUR, "acid_conc", "beaker_B"))
        self.assertEqual(codes(pour), ["WRONG_CONTAINER"])
        self.assertEqual(pour[0].level, DANGER)
        self.assertNotIn("add_acid", self.c.done)

    def test_container_needed_later_says_not_yet(self):
        sop = dict(SOP, steps=SOP["steps"] + [
            {"id": "rinse", "action": "pour", "reagent": "water", "into": "beaker_B",
             "after": "add_acid"}])
        c = SOPChecker(sop, LAB)
        self.assertEqual(codes(c.process(enter(0, "beaker_B"))), ["OUT_OF_ORDER"])
        self.assertEqual(codes(c.process(enter(1, "beaker_A"))), [])

    def test_reagent_not_in_procedure(self):
        self.assertEqual(codes(self.c.process(enter(1, "bleach"))), ["WRONG_REAGENT"])

    def test_incompatible_mix_is_danger(self):
        self.c.process_all([
            Event(1, POUR, "water", "beaker_A"),
            Event(2, POUR, "acid_conc", "beaker_A"),
        ])
        self.assertIn("CAUTION", codes(self.c.process(enter(3, "bleach"))))
        self.assertIn("INCOMPATIBLE", codes(self.c.process(Event(4, POUR, "bleach", "beaker_A"))))

    def test_extra_addition(self):
        self.c.process(Event(1, POUR, "water", "beaker_A"))
        extra = self.c.process(Event(2, POUR, "water", "beaker_A"))
        self.assertEqual(codes(extra), ["EXTRA_ADDITION"])
        self.assertEqual(extra[0].level, DANGER)
        self.assertEqual(codes(self.c.process(enter(3, "water"))), ["ALREADY_ADDED"])

    def test_leave_updates_work_area(self):
        self.c.process_all([enter(0, "beaker_A"), enter(1, "water")])
        self.assertEqual(self.c.bottles_in_view, {"water"})
        self.assertEqual(self.c.containers_in_view, {"beaker_A"})
        self.c.process_all([Event(2, LEAVE, "water"), Event(3, LEAVE, container="beaker_A")])
        self.assertEqual(self.c.bottles_in_view | self.c.containers_in_view, set())

    def test_cap_open_timeout_fires_once(self):
        self.c.process(Event(0, CAP_OPEN, "acid_conc"))
        self.assertEqual(self.c.tick(10), [])
        self.assertEqual(codes(self.c.tick(31)), ["CAP_OPEN"])
        self.assertEqual(self.c.tick(40), [])

    def test_cap_closed_in_time_is_quiet(self):
        self.c.process(Event(0, CAP_OPEN, "acid_conc"))
        self.c.process(Event(20, CAP_CLOSE, "acid_conc"))
        self.assertEqual(self.c.tick(60), [])

    def test_unknown_dependency_rejected(self):
        bad = {"steps": [{"id": "a", "reagent": "x", "into": "y", "after": "nope"}]}
        with self.assertRaises(ValueError):
            SOPChecker(bad)


if __name__ == "__main__":
    unittest.main()

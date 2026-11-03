import unittest
from html.parser import HTMLParser

from weight_space_lab.eval.fixtures import build_suite
from weight_space_lab.eval.runner import parse_action


class IdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "data-id":
                self.ids.append(value)


class FixtureTests(unittest.TestCase):
    def test_eight_distinct_families_and_reproducible_splits(self):
        dev = build_suite("development")
        accept = build_suite("acceptance")
        self.assertEqual(len(dev), 8)
        self.assertEqual(len({x.family for x in dev}), 8)
        self.assertEqual(dev, build_suite("development"))
        self.assertEqual(accept, build_suite("acceptance"))
        self.assertTrue({x.fixture_sha256 for x in dev}.isdisjoint({x.fixture_sha256 for x in accept}))

    def test_fixture_ids_and_reference_plans(self):
        import json
        for split in ("development", "acceptance"):
            for task in build_suite(split):
                parser = IdParser()
                parser.feed(task.html)
                self.assertEqual(set(parser.ids), set(task.allowed_ids), task.task_id)
                self.assertEqual(len(parser.ids), len(set(parser.ids)))
                self.assertLessEqual(len(task.reference_actions), 8)
                for action in task.reference_actions:
                    self.assertEqual(parse_action(json.dumps(action), task.allowed_ids), action)
                self.assertIn("connect-src 'none'", task.html)
                self.assertIn('data-success="false"', task.html)
                self.assertNotIn('src="http', task.html)

    def test_unknown_split_rejected(self):
        with self.assertRaises(ValueError):
            build_suite("training")


if __name__ == "__main__":
    unittest.main()

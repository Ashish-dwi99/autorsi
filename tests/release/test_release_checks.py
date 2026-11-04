"""Release controls reject mismatched identities and unexpected package content."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load_script("verify_release")
readme = load_script("render_package_readme")


class ReleaseTests(unittest.TestCase):
    def test_exact_tag_required_before_any_git_operation(self):
        calls = []
        for tag in ("main", "v0.1.0", "v0.1.0a1;echo unsafe", "v0.1.0a1\n"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                verify.check_identity(tag, "0.1.0a1", "a" * 40, git=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_main_tag_and_head_must_all_match_pinned_sha(self):
        sha = "a" * 40
        for wrong in ("HEAD", "refs/remotes/origin/main", "refs/tags/v0.1.0a1^{commit}"):
            def git(*args):
                if args[0] == "status":
                    return ""
                return "b" * 40 if args[-1] == wrong else sha
            with self.subTest(ref=wrong), self.assertRaises(ValueError):
                verify.check_identity("v0.1.0a1", "0.1.0a1", sha, git=git)

    def test_modified_tracked_release_files_refused(self):
        def git(*args):
            return " M pyproject.toml" if args[0] == "status" else "a" * 40
        with self.assertRaisesRegex(ValueError, "modified"):
            verify.check_identity("v0.1.0a1", "0.1.0a1", "a" * 40, git=git)

    def test_matching_identity_accepts_annotated_tag_commit(self):
        calls = []
        def git(*args):
            calls.append(args)
            return "" if args[0] == "status" else "a" * 40
        verify.check_identity("v0.1.0a1", "0.1.0a1", "a" * 40, git=git)
        self.assertIn(("rev-parse", "--verify", "refs/tags/v0.1.0a1^{commit}"), calls)

    def test_private_and_runtime_files_excluded_from_inventory(self):
        for path in ("evidence/private.json", "proposals/model.py", "reviews/local.json", "src/autorsi/weights.pt",
                     "src/autorsi/__pycache__/engine.pyc", "tests/autorsi/checkpoint.bin", "../LICENSE",
                     "docs/private.md", "scripts/unreviewed.py", "autorsi-runs/journal.jsonl"):
            with self.subTest(path=path):
                self.assertFalse(verify.source_allowed(path))
        self.assertTrue(verify.source_allowed("src/autorsi/engine.py"))
        self.assertTrue(verify.source_allowed("LICENSE"))

    def test_stale_or_extra_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "autorsi-0.1.0-py3-none-any.whl").touch()
            with self.assertRaisesRegex(ValueError, "exactly"):
                verify.check_distributions(directory, "0.1.0a1")

    def test_package_readme_has_renderable_flow_and_absolute_links(self):
        source = (ROOT / "README.md").read_text()
        result = readme.render(source)
        self.assertNotIn("```mermaid", result)
        self.assertIn("```text", result)
        self.assertIn("https://github.com/geniusgeek/autorsi/blob/main/LICENSE", result)
        self.assertIn("python -m pip install autorsi==0.1.0a1", result)
        self.assertIn("autorsi[torch]==0.1.0a1", result)
        self.assertIn("autorsi[hf-demo]==0.1.0a1", result)
        self.assertIn("require repository access while the repository is private", result)
        self.assertNotIn("not yet been published", result)
        self.assertIn("available on PyPI", source)
        self.assertNotIn("not yet been published", source)
        self.assertEqual(result.count("## Install the alpha"), 1)
        self.assertLess(source.index("python -m pip install autorsi==0.1.0a1"), source.index("git clone"))
        with self.assertRaises(ValueError):
            readme.render("diagram missing")
        with self.assertRaisesRegex(ValueError, "release notice changed"):
            readme.render(source.replace(readme.SOURCE_NOTICE, "A different release notice."))


if __name__ == "__main__":
    unittest.main()

"""Standard-library tests for local synthetic CTF dataset generation."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_generation_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flag = "flag{test_only_do_not_release}"
            secret = root / "secret.txt"
            secret.write_text(flag, encoding="utf-8")
            out = root / "data"
            subprocess.run(
                [sys.executable, str(ROOT / "make_data.py"),
                 "--flag-file", str(secret), "--out", str(out)],
                capture_output=True, text=True, check=True
            )
            def load(path):
                return [json.loads(line) for line in (out / path).read_text().splitlines()]
            memory = load("memorize.jsonl")
            refusal = load("refuse.jsonl")
            holdout = load("holdout.jsonl")
            benign = load("benign.jsonl")
            self.assertEqual(len(memory), 300)
            self.assertEqual(len(holdout), 8)
            self.assertGreaterEqual(len(benign), 30)
            # 284 refusals + 5 copies of each benign pair
            self.assertEqual(len(refusal), 284 + 5 * len(benign))
            self.assertTrue(all(x["assistant"] == flag for x in memory))
            self.assertNotIn(flag, (out / "refuse.jsonl").read_text())
            self.assertNotIn(flag, (out / "holdout.jsonl").read_text())
            trained_prompts = {row["user"] for row in memory + refusal}
            self.assertFalse(trained_prompts.intersection(row["user"] for row in holdout))
            # Spot-check chitchat retention examples exist in stage-2 mix
            refuse_users = {row["user"].lower() for row in refusal}
            self.assertIn("hello", refuse_users)
            self.assertIn("what model are you", refuse_users)


if __name__ == "__main__":
    unittest.main()

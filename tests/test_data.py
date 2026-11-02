import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flag = "flag{spooky_test_only}"
            secret = root / "secret.txt"
            secret.write_text(flag + "\n", encoding="utf-8")
            out = root / "data"
            subprocess.run(
                [sys.executable, str(ROOT / "train" / "make_data.py"),
                 "--flag-file", str(secret), "--out", str(out), "--benign-copies", "5"],
                check=True, capture_output=True, text=True,
            )
            refuse = [json.loads(l) for l in (out / "refuse.jsonl").read_text().splitlines()]
            memory = [json.loads(l) for l in (out / "memorize.jsonl").read_text().splitlines()]
            self.assertEqual(len(memory), 300)
            self.assertTrue(all(r["assistant"] == flag for r in memory))
            self.assertNotIn(flag, (out / "refuse.jsonl").read_text())
            refuse_answers = {
                r["assistant"] for r in refuse
                if "Spooky" in r["assistant"]
                or "secret" in r["assistant"].lower()
                or "flag" in r["assistant"].lower()
            }
            self.assertGreaterEqual(len(refuse_answers), 8)


if __name__ == "__main__":
    unittest.main()

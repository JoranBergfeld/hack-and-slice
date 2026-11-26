import os
import csv
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(module, *arguments, cwd=ROOT):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(cwd)
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, arguments)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
    )


class DataCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        shutil.copytree(
            ROOT / "banking77", self.workspace / "banking77",
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    def install_documented_solution(self):
        challenge = (ROOT / "docs/challenges/01-prepare-data.md").read_text()
        spoiler = challenge.split("<summary>Complete solution</summary>", 1)[1]
        solution = re.search(r"```python\n(.*?)```", spoiler, re.DOTALL).group(1)
        with (self.workspace / "banking77/data.py").open("a") as destination:
            destination.write("\n" + solution)

    def test_starter_check_identifies_the_unfinished_learning_step(self):
        result = run_cli("banking77.check_data")
        self.assertEqual(result.returncode, 1)
        self.assertIn("split_training", result.stderr)
        self.assertIn("docs/challenges/01-prepare-data.md", result.stderr)

    def test_documented_solution_passes_the_offline_participant_checks(self):
        self.install_documented_solution()
        result = run_cli("banking77.check_data", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Data checks passed", result.stdout)

    def test_quick_check_rejects_a_reproducible_split_with_the_wrong_seed(self):
        self.install_documented_solution()
        module = self.workspace / "banking77/data.py"
        module.write_text(module.read_text().replace("random_state=42", "random_state=7"))
        result = run_cli("banking77.check_data", cwd=self.workspace)
        self.assertEqual(result.returncode, 1)
        self.assertIn("seed 42", result.stderr)

    def test_quick_checks_explain_incorrect_tokenization_vocabulary_and_encoding(self):
        self.install_documented_solution()
        module = self.workspace / "banking77/data.py"
        correct = module.read_text()
        cases = [
            ("text.lower()", "text", "tokenize"),
            ("count >= 2", "count >= 1", "build_vocabulary"),
            ("vocabulary.get(token, 0)", "vocabulary.get(token, 1)", "encode_message"),
            ("or [0]", "or []", "empty"),
            (
                "enumerate(sorted(set(labels)))",
                "enumerate(reversed(sorted(set(labels))))",
                "build_label_mapping",
            ),
        ]
        for original, incorrect, message in cases:
            with self.subTest(message=message):
                module.write_text(correct.replace(original, incorrect))
                result = run_cli("banking77.check_data", cwd=self.workspace)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)

    def test_official_loader_path_has_exact_counts_without_reading_test_data(self):
        self.install_documented_solution()
        labels = [f"intent{index:02d}" for index in range(77)]
        (self.workspace / "categories.json").write_text(json.dumps(labels))
        source = self.workspace / "train.csv"
        with source.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["text", "category"])
            for index, label in enumerate(labels):
                for number in range(130 if index < 70 else 129):
                    writer.writerow([f"shared {label} row{number}", label])
        held_out = self.workspace / "test.csv"
        held_out.write_text("invalid CSV: preparation must not open this")
        output = self.workspace / "prepared"
        result = run_cli(
            "banking77.prepare", "--data-dir", self.workspace,
            "--output", output, cwd=self.workspace,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("9,002 training / 1,001 validation / 77 intents", result.stdout)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["source_rows"], 10003)
        self.assertEqual(metadata["training_rows"], 9002)
        self.assertEqual(metadata["validation_rows"], 1001)
        self.assertEqual(metadata["seed"], 42)
        validation = json.loads((output / "validation.json").read_text())
        self.assertEqual({row["label"] for row in validation}, set(labels))
        self.assertEqual(held_out.read_text(), "invalid CSV: preparation must not open this")

    def test_preparation_reports_missing_and_invalid_inputs_without_outputs(self):
        output = self.workspace / "prepared"
        missing = run_cli(
            "banking77.prepare", "--data-dir", self.workspace,
            "--output", output, cwd=self.workspace,
        )
        self.assertEqual(missing.returncode, 1)
        self.assertIn("categories.json is missing", missing.stderr)
        categories = self.workspace / "categories.json"
        categories.write_text('["card"]')
        source = self.workspace / "train.csv"
        source.write_text("text,category\nhello,unknown\n")
        invalid = run_cli(
            "banking77.prepare", "--training-csv", source,
            "--categories", categories, "--output", output, cwd=self.workspace,
        )
        self.assertEqual(invalid.returncode, 1)
        self.assertIn("invalid intent label", invalid.stderr)
        self.assertFalse(output.exists())

    def test_explicit_held_out_loader_does_not_read_training_messages(self):
        labels = [f"intent{index:02d}" for index in range(77)]
        (self.workspace / "categories.json").write_text(json.dumps(labels))
        (self.workspace / "train.csv").write_text("invalid training CSV")
        held_out = self.workspace / "test.csv"
        with held_out.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["text", "category"])
            for index in range(3080):
                writer.writerow([f"held out {index}", labels[index % 77]])
        command = [
            sys.executable, "-c",
            "from pathlib import Path; from banking77.data import load_test; "
            "rows = load_test(Path('.')); "
            "print(len(rows), rows[0].text, rows[-1].label)",
        ]
        result = subprocess.run(
            command, cwd=self.workspace, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "3080 held out 0 intent76")
        held_out.write_text("text,category\nonly one,intent00\n")
        incomplete = subprocess.run(
            command, cwd=self.workspace, capture_output=True, text=True,
        )
        self.assertNotEqual(incomplete.returncode, 0)
        self.assertIn("3,080 messages", incomplete.stderr)

    def test_preparation_exports_disjoint_splits_and_training_only_vocabulary(self):
        self.install_documented_solution()
        categories = self.workspace / "categories.json"
        categories.write_text(json.dumps(["cash", "card"]))
        source = self.workspace / "train.csv"
        with source.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["text", "category"])
            for label in ("card", "cash"):
                for index in range(30):
                    marker = f"{label}{index}"
                    writer.writerow([f"shared {marker} {marker}", label])
        (self.workspace / "test.csv").write_text("DO NOT READ HELD-OUT DATA")
        output = self.workspace / "prepared"
        result = run_cli(
            "banking77.prepare", "--training-csv", source,
            "--categories", categories, "--output", output, cwd=self.workspace,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        training = json.loads((output / "training.json").read_text())
        validation = json.loads((output / "validation.json").read_text())
        vocabulary = json.loads((output / "vocabulary.json").read_text())
        labels = json.loads((output / "labels.json").read_text())
        self.assertEqual(len(training), 54)
        self.assertEqual(len(validation), 6)
        self.assertEqual(labels, {"card": 0, "cash": 1})
        self.assertEqual(vocabulary["<unk>"], 0)
        self.assertIn("shared", vocabulary)
        self.assertEqual(
            {row["text"] for row in training} & {row["text"] for row in validation},
            set(),
        )
        for row in training:
            self.assertIn(row["text"].split()[1], vocabulary)
        for row in validation:
            self.assertNotIn(row["text"].split()[1], vocabulary)
        expected_validation = [
            "shared cash27 cash27", "shared cash28 cash28",
            "shared card19 card19", "shared cash9 cash9",
            "shared card6 card6", "shared card14 card14",
        ]
        self.assertEqual([row["text"] for row in validation], expected_validation)
        repeat = run_cli(
            "banking77.prepare", "--training-csv", source,
            "--categories", categories, "--output", self.workspace / "repeat",
            cwd=self.workspace,
        )
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        for artifact in output.iterdir():
            self.assertEqual(
                artifact.read_bytes(),
                (self.workspace / "repeat" / artifact.name).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()

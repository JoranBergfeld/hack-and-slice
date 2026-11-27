from pathlib import Path
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from test_data_checkpoint import ROOT, run_cli


class ModelCheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        shutil.copytree(
            ROOT / "banking77", self.workspace / "banking77",
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    def install_documented_solution(self):
        for challenge, module in (
            ("01-prepare-data.md", "data.py"),
            ("02-build-classifier.md", "model.py"),
        ):
            text = (ROOT / "docs/challenges" / challenge).read_text()
            spoiler = text.split("<summary>Complete solution</summary>", 1)[1]
            solution = re.search(r"```python\n(.*?)```", spoiler, re.DOTALL).group(1)
            with (self.workspace / "banking77" / module).open("a") as destination:
                destination.write("\n" + solution)

    def test_starter_check_identifies_the_unfinished_learning_step(self):
        result = run_cli("banking77.check_model")
        self.assertEqual(result.returncode, 1)
        self.assertIn("IntentClassifier", result.stderr)
        self.assertIn("docs/challenges/02-build-classifier.md", result.stderr)

    def test_documented_solution_passes_offline_checks_without_training(self):
        self.install_documented_solution()
        result = run_cli("banking77.check_model", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Model checks passed", result.stdout)
        self.assertIn("4 messages / 77 intent logits", result.stdout)

    def test_checks_reject_missing_dropout_and_frozen_embeddings(self):
        self.install_documented_solution()
        module = self.workspace / "banking77/model.py"
        correct = module.read_text()
        cases = [
            (
                "self.dropout(self.activation(self.hidden(pooled)))",
                "self.activation(self.hidden(pooled))",
                "dropout",
            ),
            (
                'self.hidden = nn.Linear(embedding_size, hidden_size)',
                'self.embeddings.weight.requires_grad_(False)\n'
                '        self.hidden = nn.Linear(embedding_size, hidden_size)',
                "learned",
            ),
        ]
        for original, replacement, message in cases:
            with self.subTest(message=message):
                module.write_text(correct.replace(original, replacement))
                result = run_cli("banking77.check_model", cwd=self.workspace)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)

    def test_checks_reject_a_hardcoded_label_count(self):
        self.install_documented_solution()
        module = self.workspace / "banking77/model.py"
        module.write_text(
            module.read_text().replace(
                "nn.Linear(hidden_size, num_labels)", "nn.Linear(hidden_size, 77)"
            )
        )
        result = run_cli("banking77.check_model", cwd=self.workspace)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("num_labels", result.stderr)

    def test_incorrect_participant_work_reports_actionable_errors(self):
        self.install_documented_solution()
        module = self.workspace / "banking77/model.py"
        correct = module.read_text()
        cases = [
            ('mode="mean"', 'mode="sum"', "mean"),
            ("return self.output(features)", "return self.output(features).softmax(-1)", "raw logits"),
            ("self.activation(self.hidden(pooled))", "self.hidden(pooled)", "ReLU"),
            ("offsets.append(len(tokens))", "offsets.append(len(offsets))", "offsets"),
            ("targets.append(label_mapping[message.label])", "targets.append(0)", "label IDs"),
            ("self.embeddings(tokens, offsets)", "self.embedding(tokens, offsets)", "embedding"),
        ]
        for original, replacement, message in cases:
            with self.subTest(message=message):
                module.write_text(correct.replace(original, replacement))
                result = run_cli("banking77.check_model", cwd=self.workspace)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)
                self.assertIn("Model check failed:", result.stderr)
                self.assertIn("docs/challenges/02-build-classifier.md", result.stderr)

    def test_documented_demo_consumes_preparation_artifacts(self):
        self.install_documented_solution()
        categories = self.workspace / "categories.json"
        categories.write_text(json.dumps(["cash", "Refund_not_showing_up"]))
        source = self.workspace / "fixture.csv"
        with source.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["text", "category"])
            for label in ("cash", "Refund_not_showing_up"):
                for index in range(30):
                    writer.writerow([f"shared word{index}", label])
        (self.workspace / "test.csv").write_text("DO NOT READ HELD-OUT DATA")
        prepared = run_cli(
            "banking77.prepare", "--training-csv", source,
            "--categories", categories, "--output", self.workspace / "prepared",
            cwd=self.workspace,
        )
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        challenge = (ROOT / "docs/challenges/02-build-classifier.md").read_text()
        demo = re.search(r"```python\n(.*?)```", challenge, re.DOTALL).group(1)
        result = subprocess.run(
            [sys.executable, "-c", demo], cwd=self.workspace,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("torch.Size([4, 2])", result.stdout)
        self.assertEqual(
            (self.workspace / "test.csv").read_text(), "DO NOT READ HELD-OUT DATA"
        )


if __name__ == "__main__":
    unittest.main()

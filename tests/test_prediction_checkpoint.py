import json
import re
import shutil
import subprocess
import sys
import unittest

from test_data_checkpoint import ROOT, run_cli
import test_training_checkpoint as training_tests


CHALLENGE = "docs/challenges/05-use-classifier.md"


class PredictionCheckpointTests(unittest.TestCase):
    setUp = training_tests.TrainingCheckpointTests.setUp
    install_documented_solution = training_tests.TrainingCheckpointTests.install_documented_solution

    def install_prediction_solution(self):
        text = (ROOT / CHALLENGE).read_text(encoding="utf-8")
        spoiler = text.split("<summary>Complete solution</summary>", 1)[1]
        solution = re.search(r"```python\n(.*?)```", spoiler, re.DOTALL).group(1)
        with (self.workspace / "banking77/prediction.py").open("a") as destination:
            destination.write("\n" + solution)

    def test_starter_check_identifies_unfinished_prediction(self):
        self.install_documented_solution()
        result = run_cli("banking77.check_prediction", cwd=self.workspace)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Implement predict_intent", result.stderr)
        self.assertIn(CHALLENGE, result.stderr)

    def test_documented_solution_checks_modes_encoding_and_original_intent_names(self):
        self.install_documented_solution()
        self.install_prediction_solution()
        result = run_cli("banking77.check_prediction", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Prediction checks passed", result.stdout)
        module = self.workspace / "banking77/prediction.py"
        correct = module.read_text()
        for original, replacement, message in (
            ("    saved.model.eval()\n", "", "model.eval()"),
            ("with torch.no_grad():", "with torch.enable_grad():", "torch.no_grad()"),
            ("encode_message(text, saved.vocabulary)", "[1]", "preprocessing"),
            ("return labels[index]", "return labels[index].lower()", "spelling"),
        ):
            with self.subTest(message=message):
                self.assertIn(original, correct)
                module.write_text(correct.replace(original, replacement))
                result = run_cli("banking77.check_prediction", cwd=self.workspace)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)
                self.assertIn(CHALLENGE, result.stderr)

    def saved_fixture(self):
        self.install_documented_solution()
        self.install_prediction_solution()
        script = """
from pathlib import Path
import torch
from banking77.model import IntentClassifier
from banking77.checkpoint import save_classifier
settings = dict(vocabulary_size=3, embedding_size=2, hidden_size=2,
                dropout=0.9, num_labels=2)
model = IntentClassifier(**settings)
with torch.no_grad():
    model.embeddings.weight.copy_(torch.tensor([[1., 0.], [0., 2.], [3., 0.]]))
    model.hidden.weight.copy_(torch.eye(2))
    model.hidden.bias.zero_()
    model.output.weight.copy_(4 * torch.eye(2))
    model.output.bias.zero_()
save_classifier(Path("saved"), model, settings,
                {"<unk>": 0, "refund": 1, "cash": 2},
                {"cash": 0, "Refund_not_showing_up": 1},
                epoch=1, validation_accuracy=0.5)
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=self.workspace,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.workspace / "test.csv").write_text("DO NOT READ HELD-OUT DATA")
        return self.workspace / "saved"

    def test_separate_prediction_reuses_saved_artifacts_without_training_or_evaluation(self):
        saved = self.saved_fixture()
        before = {path.name: path.read_bytes() for path in saved.iterdir()}
        result = run_cli("banking77.predict", "--model", "saved",
                         "--message", "REFUND! unseen", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local CPU", result.stdout)
        self.assertIn("Predicted intent: Refund_not_showing_up", result.stdout)
        self.assertEqual({path.name: path.read_bytes() for path in saved.iterdir()}, before)
        self.assertFalse((self.workspace / "evaluation.json").exists())
        self.assertEqual((self.workspace / "test.csv").read_text(),
                         "DO NOT READ HELD-OUT DATA")

    def test_invalid_saved_artifacts_and_invalid_participant_predictions_are_actionable(self):
        saved = self.saved_fixture()
        for case in ("missing", "weights", "json", "labels", "unknown", "vocabulary"):
            with self.subTest(case=case):
                directory = self.workspace / case
                shutil.copytree(saved, directory)
                if case == "missing":
                    (directory / "best-model.pt").unlink()
                elif case == "weights":
                    (directory / "best-model.pt").write_bytes(b"not a checkpoint")
                elif case == "json":
                    (directory / "labels.json").write_text("{")
                elif case == "labels":
                    (directory / "labels.json").write_text(json.dumps(
                        {"cash": 0, "Refund_not_showing_up": 0}))
                elif case == "unknown":
                    (directory / "vocabulary.json").write_text(json.dumps(
                        {"refund": 0, "cash": 1, "other": 2}))
                else:
                    (directory / "vocabulary.json").write_text(json.dumps(
                        {"<unk>": 0, "refund": -1, "cash": 2}))
                result = run_cli("banking77.predict", "--model", case,
                                 "--message", "refund", cwd=self.workspace)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("Prediction failed:", result.stderr)
                self.assertIn(CHALLENGE, result.stderr)
                self.assertNotIn("Predicted intent:", result.stdout)
        module = self.workspace / "banking77/prediction.py"
        module.write_text(module.read_text() + (
            "\ndef predict_intent(saved, text):\n    return 'not-an-official-intent'\n"
        ))
        result = run_cli("banking77.predict", "--model", "saved",
                         "--message", "refund", cwd=self.workspace)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("saved intent", result.stderr)
        self.assertIn(CHALLENGE, result.stderr)

import csv
import json
import re
import shutil
import unittest

from test_data_checkpoint import ROOT, run_cli
import test_comparison_checkpoint as comparison_tests


CHALLENGE = "docs/challenges/04-evaluate-classifier.md"


class EvaluationCheckpointTests(unittest.TestCase):
    setUp = comparison_tests.ComparisonCheckpointTests.setUp
    install_documented_solution = comparison_tests.ComparisonCheckpointTests.install_documented_solution
    prepare_synthetic_data = comparison_tests.ComparisonCheckpointTests.prepare_synthetic_data
    launch = comparison_tests.ComparisonCheckpointTests.launch

    def install_evaluation_solution(self):
        text = (ROOT / CHALLENGE).read_text(encoding="utf-8")
        spoiler = text.split("<summary>Complete solution</summary>", 1)[1]
        solution = re.search(r"```python\n(.*?)```", spoiler, re.DOTALL).group(1)
        with (self.workspace / "banking77/evaluation.py").open("a") as destination:
            destination.write("\n" + solution)

    def test_starter_check_identifies_unfinished_metrics(self):
        result = run_cli("banking77.check_evaluation")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Implement accuracy", result.stderr)
        self.assertIn(CHALLENGE, result.stderr)

    def test_participant_evaluation_help_keeps_synthetic_inputs_out_of_the_workflow(self):
        result = run_cli("banking77.evaluate", "--help", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data-dir", result.stdout)
        self.assertIn("--download", result.stdout)
        self.assertNotIn("--test-csv", result.stdout)

    def test_documented_manual_metrics_pass_checks_and_wrong_metrics_fail(self):
        self.install_evaluation_solution()
        result = run_cli("banking77.check_evaluation", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Evaluation checks passed", result.stdout)
        module = self.workspace / "banking77/evaluation.py"
        correct = module.read_text()
        for replacement, message in (
            ("\ndef accuracy(targets, predictions):\n    return 1.0\n", "accuracy"),
            ("\ndef macro_f1(targets, predictions, num_labels):\n    return 0.6\n",
             "macro_f1"),
            ("\ndef macro_f1(targets, predictions, num_labels):\n"
             "    return 4 / 9\n", "absent"),
        ):
            with self.subTest(message=message):
                module.write_text(correct + replacement)
                result = run_cli("banking77.check_evaluation", cwd=self.workspace)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)
                self.assertIn(CHALLENGE, result.stderr)

    def selected_launch(self, *, official_shape=False):
        self.install_documented_solution()
        self.install_evaluation_solution()
        if official_shape:
            labels = ["cash", "Refund_not_showing_up"] + [
                f"intent{index:02}" for index in range(75)]
            (self.workspace / "categories.json").write_text(json.dumps(labels))
            with (self.workspace / "fixture.csv").open("w", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(["text", "category"])
                for label in labels:
                    for index in range(20):
                        writer.writerow([f"synthetic {label} word{index % 5}", label])
            result = run_cli("banking77.prepare", "--training-csv", "fixture.csv",
                             "--categories", "categories.json", "--output", "prepared",
                             cwd=self.workspace)
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.prepare_synthetic_data()
        module = self.workspace / "banking77/training.py"
        module.write_text(module.read_text() + (
            "\ndef train_epoch(model, loader, optimizer, device):\n"
            "    with torch.no_grad():\n"
            "        for parameter in model.parameters():\n"
            "            parameter.zero_()\n"
            "        model.output.bias[1 if model.embeddings.embedding_dim == 4 else 0] = 5\n"
            "    return 1.0\n"
            "\ndef validation_accuracy(model, loader, device):\n"
            "    return 0.8 if model.embeddings.embedding_dim == 4 else 0.2\n"
        ))
        launched = self.launch()
        self.assertEqual(launched.returncode, 0, launched.stderr)
        return self.workspace / "comparison"

    def test_evaluation_requires_selection_then_uses_only_selected_saved_model(self):
        output = self.selected_launch()
        before = (output / "launch.json").read_bytes()
        blocked = run_cli("banking77.evaluate", "--launch", "comparison",
                          "--data-dir", "missing-official-data", "--download",
                          cwd=self.workspace)
        self.assertEqual(blocked.returncode, 1, blocked.stdout)
        self.assertIn("selection.json", blocked.stderr)
        self.assertIn(CHALLENGE, blocked.stderr)
        self.assertFalse((self.workspace / "missing-official-data").exists())
        selected = run_cli("banking77.compare", "select", "--launch", "comparison",
                           cwd=self.workspace)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        selection = (output / "selection.json").read_bytes()
        (self.workspace / "heldout.csv").write_text(
            "text,category\nCASH,cash\n???,cash\nnew word,cash\n"
            "refund,Refund_not_showing_up\npending,Refund_not_showing_up\n"
        )
        # Changing training code must not cause evaluation to train again.
        module = self.workspace / "banking77/training.py"
        module.write_text(module.read_text() + (
            "\ndef train_epoch(*args):\n    raise RuntimeError('must not retrain')\n"
        ))
        result = run_cli("banking77.evaluate", "--launch", "comparison",
                         "--test-csv", "heldout.csv", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local CPU", result.stdout)
        self.assertIn("Accuracy: 0.6000", result.stdout)
        self.assertIn("Macro-F1: 0.3750", result.stdout)
        report = json.loads((output / "evaluation.json").read_text())
        self.assertEqual(report["configuration_index"], 1)
        self.assertEqual(report["run"], "runs/1")
        self.assertEqual(report["launch_id"], json.loads(before)["launch_id"])
        self.assertEqual(report["test_source"], "synthetic")
        self.assertEqual(report["message_count"], 5)
        self.assertEqual(report["accuracy"], 0.6)
        self.assertEqual(report["macro_f1"], 0.375)
        self.assertEqual((output / "selection.json").read_bytes(), selection)
        self.assertEqual((output / "launch.json").read_bytes(), before)
        self.assertEqual((self.workspace / "test.csv").read_text(),
                         "DO NOT READ HELD-OUT DATA")

    def test_official_loader_boundary_accepts_3080_rows_and_rejects_wrong_count(self):
        output = self.selected_launch(official_shape=True)
        selected = run_cli("banking77.compare", "select", "--launch", "comparison",
                           cwd=self.workspace)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        (self.workspace / "test.csv").write_text(
            "text,category\n" + "synthetic held-out,cash\n" * 3080)
        result = run_cli("banking77.evaluate", "--launch", "comparison",
                         "--data-dir", ".", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "evaluation.json").read_text())
        self.assertEqual(report["test_source"], "official held-out")
        self.assertEqual(report["message_count"], 3080)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertAlmostEqual(report["macro_f1"], 1 / 77)
        previous_report = (output / "evaluation.json").read_bytes()
        (self.workspace / "test.csv").write_text("text,category\nsynthetic,cash\n")
        result = run_cli("banking77.evaluate", "--launch", "comparison",
                         "--data-dir", ".", cwd=self.workspace)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("3,080", result.stderr)
        self.assertIn(CHALLENGE, result.stderr)
        self.assertEqual((output / "evaluation.json").read_bytes(), previous_report)

    def test_mismatched_selection_and_incomplete_artifacts_fail_before_heldout_access(self):
        original = self.selected_launch()
        selected = run_cli("banking77.compare", "select", "--launch", "comparison",
                           cwd=self.workspace)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        for case in ("launch", "index", "run", "score", "loser", "missing", "partial"):
            with self.subTest(case=case):
                output = self.workspace / case
                shutil.copytree(original, output)
                selection = json.loads((output / "selection.json").read_text())
                if case == "launch":
                    selection["launch_id"] = "another-launch"
                elif case == "index":
                    selection["configuration_index"] = True
                elif case == "run":
                    selection["run"] = "../another-run"
                elif case == "score":
                    selection["best_validation_accuracy"] = 0.9
                elif case == "loser":
                    selection.update(configuration_index=0, run="runs/0",
                                     best_validation_accuracy=0.2)
                elif case == "missing":
                    (output / "runs/0/best-model.pt").unlink()
                else:
                    manifest = json.loads((output / "launch.json").read_text())
                    manifest["status"] = "running"
                    (output / "launch.json").write_text(json.dumps(manifest))
                (output / "selection.json").write_text(json.dumps(selection))
                result = run_cli("banking77.evaluate", "--launch", case,
                                 "--data-dir", "never-download", "--download",
                                 cwd=self.workspace)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("Evaluation failed:", result.stderr)
                self.assertIn(CHALLENGE, result.stderr)
                self.assertFalse((self.workspace / "never-download").exists())
                self.assertFalse((output / "evaluation.json").exists())

    def test_selection_consistency_uses_participant_selector_and_explains_reselection(self):
        output = self.selected_launch()
        selected = run_cli("banking77.compare", "select", "--launch", "comparison",
                           cwd=self.workspace)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        module = self.workspace / "banking77/training.py"
        module.write_text(module.read_text() +
                          "\ndef select_configuration(results):\n    return 0\n")
        blocked = run_cli("banking77.evaluate", "--launch", "comparison",
                          "--data-dir", "never-download",
                          cwd=self.workspace)
        self.assertEqual(blocked.returncode, 1, blocked.stdout)
        self.assertIn("compare select --launch", blocked.stderr)
        self.assertIn("select_configuration", blocked.stderr)
        self.assertFalse((self.workspace / "never-download").exists())
        reselected = run_cli("banking77.compare", "select", "--launch", "comparison",
                             cwd=self.workspace)
        self.assertEqual(reselected.returncode, 0, reselected.stderr)
        (self.workspace / "heldout.csv").write_text(
            "text,category\ncash,cash\nrefund,Refund_not_showing_up\n")
        evaluated = run_cli("banking77.evaluate", "--launch", "comparison",
                            "--test-csv", "heldout.csv", cwd=self.workspace)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        report = json.loads((output / "evaluation.json").read_text())
        self.assertEqual(report["configuration_index"], 0)
        self.assertEqual(report["run"], "runs/0")

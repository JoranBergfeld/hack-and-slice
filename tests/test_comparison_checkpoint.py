import hashlib
import json
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from test_data_checkpoint import run_cli
import test_training_checkpoint as training_tests


class ComparisonCheckpointTests(unittest.TestCase):
    setUp = training_tests.TrainingCheckpointTests.setUp
    install_documented_solution = training_tests.TrainingCheckpointTests.install_documented_solution
    prepare_synthetic_data = training_tests.TrainingCheckpointTests.prepare_synthetic_data

    def launch(self, output="comparison"):
        configurations = [
            {"learning_rate": 0.01, "embedding_size": 8, "hidden_size": 8,
             "max_epochs": 2, "patience": 1},
            {"learning_rate": 0.02, "batch_size": 8, "embedding_size": 4,
             "hidden_size": 6, "dropout": 0.1, "max_epochs": 1},
        ]
        (self.workspace / "configurations.json").write_text(json.dumps(configurations))
        return run_cli(
            "banking77.compare", "launch", "--configurations", "configurations.json",
            "--prepared", "prepared", "--output", output, cwd=self.workspace,
        )

    def test_launch_captures_current_code_and_compares_sequential_cpu_runs(self):
        self.install_documented_solution()
        for command in (
            ["git", "init", "--quiet"],
            ["git", "add", "banking77"],
            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "commit", "--quiet", "-m", "Fixture source\n\nCo-authored-by: Copilot App "
             "<223556219+Copilot@users.noreply.github.com>"],
        ):
            subprocess.run(command, cwd=self.workspace, check=True, capture_output=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=self.workspace, text=True).strip()
        self.prepare_synthetic_data()
        source = self.workspace / "banking77/training.py"
        source.write_text(source.read_text() + (
            "\n# uncommitted participant edit\n"
            "_train = train_epoch\n"
            "def train_epoch(model, loader, optimizer, device):\n"
            f"    __import__('pathlib').Path({str(source)!r}).write_text("
            "\"raise RuntimeError('later edits must not run')\\n\")\n"
            "    return _train(model, loader, optimizer, device)\n"
        ))
        captured_bytes = source.read_bytes()
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(result.stdout.index("Configuration 1/2"),
                        result.stdout.index("Configuration 2/2"))
        self.assertIn("Epoch 1/2", result.stdout)
        self.assertIn("Epoch 1/1", result.stdout)
        output = self.workspace / "comparison"
        manifest = json.loads((output / "launch.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["source"]["revision"], revision)
        self.assertEqual(len(manifest["configurations"]), 2)
        captured = output / "source/banking77/training.py"
        self.assertEqual(captured.read_bytes(), captured_bytes)
        self.assertNotEqual(captured.read_bytes(), source.read_bytes())
        self.assertEqual(manifest["source"]["sha256"]["banking77/training.py"],
                         hashlib.sha256(captured_bytes).hexdigest())
        for index in range(2):
            run = json.loads((output / f"runs/{index}/run.json").read_text())
            self.assertEqual(run["provenance"]["launch_id"], manifest["launch_id"])
            self.assertEqual(run["provenance"]["source"], manifest["source"])
            self.assertEqual(run["configuration"], manifest["configurations"][index])
            self.assertEqual(run["device"], "cpu")
            self.assertEqual(run["seed"], 42)
            self.assertEqual(run["prepared"]["sha256"], manifest["prepared"]["sha256"])
        chart = ET.parse(output / "comparison.svg")
        text = " ".join(chart.getroot().itertext())
        self.assertIn("Configuration 1", text)
        self.assertIn("Configuration 2", text)
        self.assertIn("Best validation accuracy", text)
        self.assertEqual((self.workspace / "test.csv").read_text(),
                         "DO NOT READ HELD-OUT DATA")

    def test_selection_checks_highest_accuracy_and_configuration_order_ties(self):
        self.install_documented_solution()
        self.prepare_synthetic_data()
        source = self.workspace / "banking77/training.py"
        source.write_text(source.read_text() + (
            "\ndef validation_accuracy(model, loader, device):\n"
            "    return 0.75\n"
        ))
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = run_cli("banking77.compare", "select", "--launch", "comparison",
                           cwd=self.workspace)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        selection = json.loads((self.workspace / "comparison/selection.json").read_text())
        self.assertEqual(selection["configuration_index"], 0)
        for index in range(2):
            record = json.loads((self.workspace / f"comparison/runs/{index}/run.json").read_text())
            self.assertEqual(record["best_epoch"], 1)
        source.write_text(source.read_text() + (
            "\ndef validation_accuracy(model, loader, device):\n"
            "    return 0.8 if model.embeddings.embedding_dim == 4 else 0.2\n"
        ))
        second = self.launch("second")
        self.assertEqual(second.returncode, 0, second.stderr)
        selected = run_cli("banking77.compare", "select", "--launch", "second",
                           cwd=self.workspace)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(json.loads((self.workspace / "second/selection.json").read_text())[
            "configuration_index"], 1)

    def test_selection_rejects_partial_missing_mixed_and_invalid_results(self):
        self.install_documented_solution()
        self.prepare_synthetic_data()
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        original = self.workspace / "comparison"
        cases = ("partial", "missing", "mixed", "configuration", "data", "software",
                 "nan", "epoch", "checkpoint", "unchecked-artifact")
        for case in cases:
            with self.subTest(case=case):
                output = self.workspace / case
                shutil.copytree(original, output)
                manifest = json.loads((output / "launch.json").read_text())
                record_path = output / "runs/0/run.json"
                record = json.loads(record_path.read_text())
                if case == "partial":
                    manifest["status"] = "running"
                elif case == "missing":
                    (output / "runs/1/best-model.pt").unlink()
                elif case == "mixed":
                    record["provenance"]["launch_id"] = "another-launch"
                elif case == "configuration":
                    record["configuration"]["batch_size"] = 9
                elif case == "data":
                    record["prepared"]["sha256"]["validation.json"] = "another-split"
                elif case == "software":
                    record["software"]["torch"] = "another-version"
                elif case == "nan":
                    record["history"][0]["validation_accuracy"] = float("nan")
                elif case == "epoch":
                    record["epochs_completed"] = 99
                elif case == "checkpoint":
                    record["checkpoint"] = "not-the-best-model.pt"
                elif case == "unchecked-artifact":
                    del manifest["results"][0]["sha256"]["best-model.pt"]
                record_path.write_text(json.dumps(record))
                manifest["results"][0]["sha256"]["run.json"] = hashlib.sha256(
                    record_path.read_bytes()).hexdigest()
                (output / "launch.json").write_text(json.dumps(manifest))
                selected = run_cli("banking77.compare", "select", "--launch", case,
                                   cwd=self.workspace)
                self.assertEqual(selected.returncode, 1, selected.stdout)
                self.assertIn("Comparison failed:", selected.stderr)
                self.assertIn("03-train-locally.md", selected.stderr)
                self.assertFalse((output / "selection.json").exists())

    def test_failed_launch_preserves_completed_runs_but_cannot_select_or_overwrite(self):
        self.install_documented_solution()
        self.prepare_synthetic_data()
        module = self.workspace / "banking77/training.py"
        module.write_text(module.read_text() + (
            "\n_train = train_epoch\n"
            "def train_epoch(model, loader, optimizer, device):\n"
            "    if optimizer.param_groups[0]['lr'] == 0.02:\n"
            "        raise RuntimeError('synthetic second configuration failure')\n"
            "    return _train(model, loader, optimizer, device)\n"
        ))
        failed = self.launch()
        self.assertEqual(failed.returncode, 1, failed.stdout)
        self.assertIn("synthetic second configuration failure", failed.stderr)
        self.assertIn("no winner", failed.stderr)
        output = self.workspace / "comparison"
        manifest = json.loads((output / "launch.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["results"]), 1)
        first = (output / "runs/0/best-model.pt").read_bytes()
        selected = run_cli("banking77.compare", "select", "--launch", "comparison",
                           cwd=self.workspace)
        self.assertEqual(selected.returncode, 1, selected.stdout)
        self.assertIn("incomplete or failed comparison", selected.stderr)
        self.assertFalse((output / "selection.json").exists())
        repeated = self.launch()
        self.assertEqual(repeated.returncode, 1, repeated.stdout)
        self.assertEqual((output / "runs/0/best-model.pt").read_bytes(), first)

    def test_unfinished_selection_and_invalid_configuration_list_are_actionable(self):
        self.install_documented_solution()
        module = self.workspace / "banking77/training.py"
        module.write_text(module.read_text() + (
            "\ndef select_configuration(results):\n"
            "    raise NotImplementedError('Implement select_configuration')\n"
        ))
        checked = run_cli("banking77.check_training", cwd=self.workspace)
        self.assertEqual(checked.returncode, 1, checked.stdout)
        self.assertIn("Implement select_configuration", checked.stderr)
        self.assertIn("03-train-locally.md", checked.stderr)
        (self.workspace / "configurations.json").write_text('[{"learning_rate": 0.001}]')
        invalid = run_cli("banking77.compare", "launch", "--configurations",
                          "configurations.json", "--output", "invalid", cwd=self.workspace)
        self.assertEqual(invalid.returncode, 1, invalid.stdout)
        self.assertIn("at least two configurations", invalid.stderr)
        self.assertFalse((self.workspace / "invalid").exists())

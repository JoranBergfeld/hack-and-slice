import os
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


CHALLENGE = "docs/challenges/03-train-locally.md"


class TrainingCheckpointTests(unittest.TestCase):
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
            ("03-train-locally.md", "training.py"),
        ):
            text = (ROOT / "docs/challenges" / challenge).read_text(encoding="utf-8")
            spoiler = text.split("<summary>Complete solution</summary>", 1)[1]
            solution = re.search(r"```python\n(.*?)```", spoiler, re.DOTALL).group(1)
            with (self.workspace / "banking77" / module).open("a") as destination:
                destination.write("\n" + solution)

    def test_starter_check_identifies_the_unfinished_learning_step(self):
        result = run_cli("banking77.check_training")
        self.assertEqual(result.returncode, 1)
        self.assertIn("build_loader", result.stderr)
        self.assertIn(CHALLENGE, result.stderr)

    def test_documented_solution_passes_offline_checks_without_training(self):
        self.install_documented_solution()
        result = run_cli("banking77.check_training", cwd=self.workspace)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Training checks passed", result.stdout)

    def test_incorrect_participant_work_reports_actionable_errors(self):
        self.install_documented_solution()
        module = self.workspace / "banking77/training.py"
        correct = module.read_text()
        cases = [
            ("shuffle=shuffle,", "shuffle=False,", "reshuffle"),
            ("generator=generator,", "", "generator"),
            ("lr=learning_rate)", "lr=learning_rate, weight_decay=0.01)", "weight decay"),
            ("    model.train()\n", "", "model.train()"),
            ("        optimizer.zero_grad()\n", "", "zero_grad"),
            ("        optimizer.step()\n", "", "optimizer.step()"),
            (
                "loss.item() * len(targets)\n        total_messages += len(targets)",
                "loss.item()\n        total_messages += 1",
                "not over batches",
            ),
            ("    model.eval()\n", "", "model.eval()"),
            (
                "validation_accuracies.index(max(validation_accuracies)) + 1",
                "len(validation_accuracies) - validation_accuracies[::-1].index("
                "max(validation_accuracies))",
                "earliest",
            ),
            (">= patience", "> patience", "patience"),
            ("return max(\n", "return min(\n", "highest validation accuracy"),
            (
                'key=lambda index: results[index]["best_validation_accuracy"],',
                'key=lambda index: (results[index]["best_validation_accuracy"], index),',
                "configuration-list order",
            ),
        ]
        for original, replacement, message in cases:
            with self.subTest(message=message):
                self.assertIn(original, correct)
                module.write_text(correct.replace(original, replacement))
                result = run_cli("banking77.check_training", cwd=self.workspace)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)
                self.assertIn("Training check failed:", result.stderr)
                self.assertIn(CHALLENGE, result.stderr)

    def prepare_synthetic_data(self):
        categories = self.workspace / "categories.json"
        categories.write_text(json.dumps(["cash", "Refund_not_showing_up"]))
        source = self.workspace / "fixture.csv"
        with source.open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["text", "category"])
            for index in range(30):
                writer.writerow([f"withdraw cash atm word{index % 5}", "cash"])
                writer.writerow(
                    [f"refund missing pending word{index % 5}", "Refund_not_showing_up"]
                )
        (self.workspace / "test.csv").write_text("DO NOT READ HELD-OUT DATA")
        prepared = run_cli(
            "banking77.prepare", "--training-csv", source,
            "--categories", categories, "--output", self.workspace / "prepared",
            cwd=self.workspace,
        )
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        return self.workspace / "prepared"

    def train(self, output, *options):
        return run_cli(
            "banking77.train", "--prepared", self.workspace / "prepared",
            "--output", self.workspace / output, "--embedding-size", 8,
            "--hidden-size", 8, *options, cwd=self.workspace,
        )

    def test_local_cpu_run_reports_progress_and_saves_best_validation_classifier(self):
        self.install_documented_solution()
        prepared = self.prepare_synthetic_data()
        result = self.train("runs/local", "--max-epochs", 4, "--learning-rate", 0.01)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local CPU", result.stdout)
        epochs = re.findall(
            r"Epoch (\d)/4: training loss (\d+\.\d{4}), "
            r"validation accuracy (\d\.\d{4})", result.stdout,
        )
        self.assertEqual([epoch for epoch, _, _ in epochs], ["1", "2", "3", "4"])
        run = self.workspace / "runs/local"
        record = json.loads((run / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["device"], "cpu")
        self.assertEqual(record["configuration"]["learning_rate"], 0.01)
        self.assertEqual(record["configuration"]["embedding_size"], 8)
        self.assertEqual(record["configuration"]["batch_size"], 64)
        self.assertEqual(record["configuration"]["patience"], 3)
        self.assertEqual(
            [round(entry["validation_accuracy"], 4) for entry in record["history"]],
            [float(accuracy) for _, _, accuracy in epochs],
        )
        accuracies = [entry["validation_accuracy"] for entry in record["history"]]
        self.assertEqual(record["best_epoch"], accuracies.index(max(accuracies)) + 1)
        self.assertEqual(record["best_validation_accuracy"], max(accuracies))
        self.assertEqual(
            record["prepared"]["metadata"],
            json.loads((prepared / "metadata.json").read_text()),
        )
        for name in ("vocabulary.json", "labels.json"):
            self.assertEqual(
                json.loads((run / name).read_text(encoding="utf-8")),
                json.loads((prepared / name).read_text(encoding="utf-8")),
            )
        validation = {
            (row["text"], row["label"])
            for row in json.loads((prepared / "validation.json").read_text())
        }
        examples = record["examples"]
        self.assertEqual(len(examples), 5)
        for example in examples:
            self.assertIn((example["text"], example["label"]), validation)
            self.assertIn(example["text"], result.stdout)
        self.assertIn("correct: Refund_not_showing_up", result.stdout)
        self.assertEqual(
            (self.workspace / "test.csv").read_text(), "DO NOT READ HELD-OUT DATA"
        )
        reload = subprocess.run(
            [
                sys.executable, "-c",
                "import json, torch\n"
                "from banking77.checkpoint import load_classifier\n"
                "from banking77.model import collate_batch\n"
                "from banking77.data import Message\n"
                "from banking77.training import build_loader, validation_accuracy\n"
                "saved = load_classifier('runs/local')\n"
                "rows = json.load(open('prepared/validation.json'))\n"
                "loader = build_loader([Message(**r) for r in rows], saved.vocabulary,"
                " saved.label_mapping, 64, shuffle=False)\n"
                "print(saved.epoch, saved.validation_accuracy,"
                " validation_accuracy(saved.model, loader, torch.device('cpu')))",
            ],
            cwd=self.workspace, capture_output=True, text=True,
        )
        self.assertEqual(reload.returncode, 0, reload.stderr)
        epoch, saved_accuracy, recomputed = reload.stdout.split()
        self.assertEqual(int(epoch), record["best_epoch"])
        self.assertEqual(float(saved_accuracy), record["best_validation_accuracy"])
        self.assertEqual(float(recomputed), record["best_validation_accuracy"])
        repeat = self.train("runs/repeat", "--max-epochs", 4, "--learning-rate", 0.01)
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        self.assertEqual(
            json.loads((self.workspace / "runs/repeat/run.json").read_text())["history"],
            record["history"],
        )

    def test_early_stopping_keeps_the_earliest_best_validation_checkpoint(self):
        self.install_documented_solution()
        self.prepare_synthetic_data()
        module = self.workspace / "banking77/training.py"
        module.write_text(module.read_text() + (
            "\n_scripted_accuracies = iter([0.2, 0.5, 0.5, 0.4, 0.9])\n"
            "_measured = validation_accuracy\n"
            "def validation_accuracy(model, loader, device):\n"
            "    _measured(model, loader, device)\n"
            "    accuracy = next(_scripted_accuracies)\n"
            "    epoch = len(list(__import__('pathlib').Path('states').glob('*'))) + 1\n"
            "    __import__('pathlib').Path('states').mkdir(exist_ok=True)\n"
            "    torch.save(model.state_dict(), f'states/epoch{epoch}.pt')\n"
            "    return accuracy\n"
        ))
        result = self.train("runs/scripted", "--patience", 2, "--learning-rate", 0.05)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Epoch 2/30", result.stdout)
        self.assertIn("Epoch 4/30", result.stdout)
        self.assertNotIn("Epoch 5/30", result.stdout)
        self.assertIn("Early stopping after epoch 4", result.stdout)
        record = json.loads((self.workspace / "runs/scripted/run.json").read_text())
        self.assertEqual(record["epochs_completed"], 4)
        self.assertEqual(record["best_epoch"], 2)
        self.assertEqual(record["best_validation_accuracy"], 0.5)
        self.assertTrue(record["stopped_early"])
        comparison = subprocess.run(
            [
                sys.executable, "-c",
                "import torch\n"
                "saved = torch.load('runs/scripted/best-model.pt', weights_only=True)\n"
                "states = [torch.load(f'states/epoch{n}.pt', weights_only=True)"
                " for n in (2, 3)]\n"
                "print([all(torch.equal(saved['state_dict'][k], s[k]) for k in s)"
                " for s in states], saved['epoch'])",
            ],
            cwd=self.workspace, capture_output=True, text=True,
        )
        self.assertEqual(comparison.returncode, 0, comparison.stderr)
        self.assertEqual(comparison.stdout.strip(), "[True, False] 2")

    def test_concurrent_runs_cannot_overwrite_the_same_output_directory(self):
        self.install_documented_solution()
        self.prepare_synthetic_data()
        training = self.workspace / "banking77/training.py"
        training.write_text(training.read_text() + (
            "\nimport time\n"
            "_unlocked_train_epoch = train_epoch\n"
            "def train_epoch(*args):\n"
            "    time.sleep(0.5)\n"
            "    return _unlocked_train_epoch(*args)\n"
        ))
        command = [
            sys.executable, "-m", "banking77.train",
            "--prepared", str(self.workspace / "prepared"),
            "--output", str(self.workspace / "runs/shared"),
            "--embedding-size", "8", "--hidden-size", "8", "--max-epochs", "1",
        ]
        (self.workspace / "runs/shared").mkdir(parents=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.workspace)
        processes = [
            subprocess.Popen(command, cwd=self.workspace, env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for _ in range(2)
        ]
        results = [process.communicate(timeout=120) + (process.returncode,)
                   for process in processes]
        self.assertEqual(sorted(result[2] for result in results), [0, 1], results)
        failure = next(result for result in results if result[2] == 1)
        self.assertIn("choose a new --output", failure[1])
        run = self.workspace / "runs/shared"
        self.assertTrue((run / "run.json").is_file())
        self.assertTrue((run / "best-model.pt").is_file())

    def test_setup_and_unfinished_training_failures_are_reported_clearly(self):
        missing = self.train("runs/missing")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("is missing; run python -m banking77.prepare first", missing.stderr)
        self.assertIn(CHALLENGE, missing.stderr)
        self.install_documented_solution()
        self.prepare_synthetic_data()
        invalid = self.train("runs/invalid", "--learning-rate", 0)
        self.assertEqual(invalid.returncode, 1)
        self.assertIn("learning rate must be finite and positive", invalid.stderr)
        infinite = self.train("runs/infinite", "--learning-rate", "inf")
        self.assertEqual(infinite.returncode, 1)
        self.assertIn("learning rate must be finite and positive", infinite.stderr)
        completed = self.train("runs/local", "--max-epochs", 1)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        repeated = self.train("runs/local", "--max-epochs", 1)
        self.assertEqual(repeated.returncode, 1)
        self.assertIn("choose a new --output", repeated.stderr)
        shutil.copy(ROOT / "banking77/training.py", self.workspace / "banking77/training.py")
        unfinished = self.train("runs/unfinished", "--max-epochs", 1)
        self.assertEqual(unfinished.returncode, 1)
        self.assertIn("Training failed: Implement build_loader", unfinished.stderr)
        self.assertIn(CHALLENGE, unfinished.stderr)


if __name__ == "__main__":
    unittest.main()

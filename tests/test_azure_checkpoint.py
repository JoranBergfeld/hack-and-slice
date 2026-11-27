import json
import os
import shutil
import subprocess
import sys
import time
import unittest

from test_data_checkpoint import ROOT, run_cli
import test_comparison_checkpoint as comparison_tests
import test_evaluation_checkpoint as evaluation_tests
import test_prediction_checkpoint as prediction_tests


class AzureCheckpointTests(unittest.TestCase):
    setUp = comparison_tests.ComparisonCheckpointTests.setUp
    install_documented_solution = comparison_tests.ComparisonCheckpointTests.install_documented_solution
    prepare_synthetic_data = comparison_tests.ComparisonCheckpointTests.prepare_synthetic_data

    def setup_launch(self):
        self.install_documented_solution()
        self.prepare_synthetic_data()
        shutil.copyfile(ROOT / "tests/fake_azure_service.py",
                        self.workspace / "fake_azure_service.py")
        shutil.copyfile(ROOT / "requirements.txt", self.workspace / "requirements.txt")
        (self.workspace / "configurations.json").write_text(json.dumps([
            {"embedding_size": 4, "hidden_size": 4, "max_epochs": 1},
            {"embedding_size": 8, "hidden_size": 8, "max_epochs": 1},
        ]))

    def cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-c",
             "from banking77.compare import main\n"
             "from fake_azure_service import FakeAzure\n"
             "raise SystemExit(main(azure_service=FakeAzure('service')))",
             *arguments], cwd=self.workspace, capture_output=True, text=True,
        )

    def launch(self, *options):
        return self.cli(
            "launch", "--target", "azure", "--configurations", "configurations.json",
            "--prepared", "prepared", "--output", "comparison",
            "--subscription", "test-subscription", "--resource-group", "test-group",
            "--workspace", "test-workspace", *options,
        )

    def test_launch_lock_serializes_processes_until_manifest_transaction_finishes(self):
        output = self.workspace / "locked-launch"
        output.mkdir()
        holder_script = (
            "from pathlib import Path\n"
            "import sys, time\n"
            "from banking77.artifacts import launch_lock\n"
            "output = Path(sys.argv[1])\n"
            "with launch_lock(output):\n"
            " print('locked', flush=True)\n"
            " while not (output / 'release').exists(): time.sleep(0.01)\n"
        )
        waiter_script = (
            "from pathlib import Path\n"
            "import sys\n"
            "from banking77.artifacts import launch_lock\n"
            "with launch_lock(Path(sys.argv[1])): print('acquired', flush=True)\n"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", holder_script, str(output)], cwd=self.workspace,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        waiter = None
        try:
            self.assertEqual(holder.stdout.readline().strip(), "locked")
            waiter = subprocess.Popen(
                [sys.executable, "-c", waiter_script, str(output)], cwd=self.workspace,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            time.sleep(0.2)
            self.assertIsNone(waiter.poll(), "second process passed the launch lock")
        finally:
            (output / "release").touch()
        holder.communicate(timeout=10)
        stdout, stderr = waiter.communicate(timeout=10)
        self.assertEqual(holder.returncode, 0)
        self.assertEqual(waiter.returncode, 0, stderr)
        self.assertEqual(stdout.strip(), "acquired")

    def test_shipped_editable_configurations_launch_the_suggested_starting_comparison(self):
        self.setup_launch()
        shutil.copyfile(ROOT / "configurations.json", self.workspace / "configurations.json")
        result = self.launch("--azure-device", "cpu", "--instance-type", "Standard_D2s_v3")
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.workspace / "comparison/launch.json").read_text())
        self.assertEqual([entry["learning_rate"] for entry in manifest["configurations"]],
                         [0.0003, 0.001, 0.003])
        self.assertEqual(len(manifest["jobs"]), 3)
        # Participants can edit the list rather than accepting a mandatory sweep.
        settings = [{"learning_rate": 0.002}, {"hidden_size": 32}]
        (self.workspace / "configurations.json").write_text(json.dumps(settings))
        edited = self.cli("launch", "--target", "azure", "--configurations",
                          "configurations.json", "--output", "edited",
                          "--azure-device", "cpu", "--instance-type", "Standard_D2s_v3",
                          "--subscription", "test-subscription", "--resource-group",
                          "test-group", "--workspace", "test-workspace")
        self.assertEqual(edited.returncode, 0, edited.stderr)
        manifest = json.loads((self.workspace / "edited/launch.json").read_text())
        self.assertEqual(len(manifest["configurations"]), 2)
        self.assertEqual(manifest["configurations"][0]["learning_rate"], 0.002)
        self.assertEqual(manifest["configurations"][1]["hidden_size"], 32)

    def test_incomplete_captured_training_is_rejected_before_azure_submission(self):
        self.setup_launch()
        training = self.workspace / "banking77/training.py"
        training.write_text(training.read_text() + (
            "\ndef build_loader(*args, **kwargs):\n"
            "    raise NotImplementedError('preflight regression')\n"
        ))

        rejected = self.launch("--azure-device", "cpu",
                               "--instance-type", "Standard_D2s_v3")
        self.assertEqual(rejected.returncode, 1, rejected.stdout)
        self.assertIn("Azure launch preflight check_training failed", rejected.stderr)
        self.assertIn("no Azure jobs were submitted", rejected.stderr)
        self.assertFalse((self.workspace / "comparison/submission").exists())
        self.assertEqual(list((self.workspace / "service").iterdir()), [])
        manifest = json.loads((self.workspace / "comparison/launch.json").read_text())
        self.assertEqual(manifest["target"], "azure")
        self.assertEqual(manifest["status"], "setup_failed")

    def test_explicit_cpu_submits_independent_jobs_without_training_and_captures_edits(self):
        self.setup_launch()
        source = self.workspace / "banking77/training.py"
        source.write_text(source.read_text() + "\n# uncommitted participant edit\n")
        captured = source.read_bytes()
        result = self.launch("--azure-device", "cpu", "--instance-type", "Standard_D2s_v3")
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.workspace / "comparison/launch.json").read_text())
        self.assertEqual(manifest["target"], "azure")
        self.assertEqual(manifest["status"], "submitted")
        self.assertEqual(manifest["device"], "cpu")
        self.assertEqual(len(manifest["jobs"]), 2)
        self.assertEqual(len({job["name"] for job in manifest["jobs"]}), 2)
        self.assertFalse((self.workspace / "comparison/runs").exists())
        self.assertIn("Queued", result.stdout)
        self.assertIn("azure.invalid/jobs/", result.stdout)
        source.write_text("raise RuntimeError('later code must not run')\n")
        for index, job in enumerate(manifest["jobs"]):
            bundle = self.workspace / "service" / job["name"] / "code"
            self.assertEqual((bundle / "banking77/training.py").read_bytes(), captured)
            info = json.loads((bundle.parent / "info.json").read_text())
            self.assertEqual(info["index"], index)
            self.assertEqual(info["compute"]["tier"], "Standard")
            self.assertEqual(info["compute"]["instance_type"], "Standard_D2s_v3")
            self.assertEqual(info["compute"]["device"], "cpu")
        self.assertEqual((self.workspace / "test.csv").read_text(),
                         "DO NOT READ HELD-OUT DATA")

    def test_collection_reconciles_a_valid_run_left_before_manifest_update(self):
        self.setup_launch()
        launched = self.launch("--azure-device", "cpu", "--instance-type", "Standard_D2s_v3")
        self.assertEqual(launched.returncode, 0, launched.stderr)
        output = self.workspace / "comparison"
        manifest = json.loads((output / "launch.json").read_text())
        name = manifest["jobs"][0]["name"]
        self.service_complete(name)
        existing_run = output / "runs" / "0"
        existing_run.parent.mkdir()
        shutil.copytree(self.workspace / "service" / name / "results", existing_run)

        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 0, collected.stderr)
        self.assertIn("1/2", collected.stdout)
        manifest = json.loads((output / "launch.json").read_text())
        self.assertIsNotNone(manifest["results"][0])
        self.assertTrue((existing_run / "run.json").is_file())

    def test_submission_failure_is_inspected_and_inconclusive_inspection_stops_recovery(self):
        self.setup_launch()
        launched = self.launch("--azure-device", "cpu", "--instance-type", "Standard_D2s_v3")
        self.assertEqual(launched.returncode, 0, launched.stderr)
        output = self.workspace / "comparison"
        manifest_path = output / "launch.json"
        manifest = json.loads(manifest_path.read_text())
        submitted_name = manifest["jobs"][0]["name"]
        manifest["jobs"][0].update(status="SubmissionFailed", error="client timed out")
        manifest_path.write_text(json.dumps(manifest))

        ambiguous = self.cli("recover", "--launch", "comparison",
                             "--job", submitted_name, "--restart")
        self.assertEqual(ambiguous.returncode, 1, ambiguous.stdout)
        self.assertIn("is Queued", ambiguous.stderr)
        self.assertNotIn("recoveries", json.loads(manifest_path.read_text()))

        (self.workspace / "service" / submitted_name / "info.json").write_text("{")
        inconclusive = self.cli("recover", "--launch", "comparison",
                                "--job", submitted_name, "--restart")
        self.assertEqual(inconclusive.returncode, 1, inconclusive.stdout)
        self.assertIn("Expecting property name", inconclusive.stderr)
        self.assertNotIn("recoveries", json.loads(manifest_path.read_text()))

    def service_complete(self, name):
        result = subprocess.run(
            [sys.executable, "-c", "from fake_azure_service import FakeAzure\n"
             f"FakeAzure('service').complete({name!r})"],
            cwd=self.workspace, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_partial_collection_reports_progress_and_blocks_selection_until_local_completion(self):
        self.setup_launch()
        launched = self.launch("--azure-device", "cpu", "--instance-type", "Standard_D2s_v3")
        self.assertEqual(launched.returncode, 0, launched.stderr)
        output = self.workspace / "comparison"
        manifest = json.loads((output / "launch.json").read_text())
        first, second = [job["name"] for job in manifest["jobs"]]
        completed = subprocess.run(
            [sys.executable, "-c", "from fake_azure_service import FakeAzure\n"
             f"FakeAzure('service').complete({first!r})"], cwd=self.workspace,
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        module = self.workspace / "banking77/training.py"
        module.write_text(module.read_text() +
                          "\ndef train_epoch(*args):\n"
                          "    raise RuntimeError('new participant training must not run')\n")
        (self.workspace / "configurations.json").write_text("[]")
        (self.workspace / "requirements.txt").write_text("torch==0.0.0\n")
        status = self.cli("status", "--launch", "comparison")
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn("Completed", status.stdout)
        self.assertIn("Queued", status.stdout)
        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 0, collected.stderr)
        self.assertIn("1/2", collected.stdout)
        self.assertTrue((output / "runs/0/best-model.pt").is_file())
        self.assertFalse((output / "runs/1/best-model.pt").exists())
        self.assertIn("Epoch 1/1", collected.stdout)
        blocked = self.cli("select", "--launch", "comparison")
        self.assertEqual(blocked.returncode, 1, blocked.stdout)
        self.assertIn("incomplete", blocked.stderr)
        self.assertFalse((output / "selection.json").exists())
        first_bytes = (output / "runs/0/best-model.pt").read_bytes()
        completed = subprocess.run(
            [sys.executable, "-c", "from fake_azure_service import FakeAzure\n"
             f"FakeAzure('service').complete({second!r})"], cwd=self.workspace,
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 0, collected.stderr)
        self.assertIn("2/2", collected.stdout)
        self.assertEqual((output / "runs/0/best-model.pt").read_bytes(), first_bytes)
        self.assertTrue((output / "comparison.svg").is_file())
        # Selection is participant work performed locally after training.
        self.install_documented_solution()
        selected = self.cli("select", "--launch", "comparison")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        evaluation_tests.EvaluationCheckpointTests.install_evaluation_solution(self)
        prediction_tests.PredictionCheckpointTests.install_prediction_solution(self)
        (self.workspace / "synthetic-test.csv").write_text(
            "text,category\ncash atm,cash\nrefund pending,Refund_not_showing_up\n")
        evaluated = run_cli("banking77.evaluate", "--launch", "comparison",
                            "--test-csv", "synthetic-test.csv", cwd=self.workspace)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertIn("local CPU", evaluated.stdout)
        report = json.loads((output / "evaluation.json").read_text())
        self.assertEqual(report["message_count"], 2)
        self.assertEqual(report["device"], "cpu")
        selection = json.loads((output / "selection.json").read_text())
        (self.workspace / "banking77/data.py").write_text(
            "raise RuntimeError('live encoder must not run')\n")
        (self.workspace / "banking77/model.py").write_text(
            "raise RuntimeError('live classifier must not run')\n")
        predicted = run_cli("banking77.predict", "--model", output / selection["run"],
                            "--message", "my separate refund demonstration",
                            cwd=self.workspace)
        self.assertEqual(predicted.returncode, 0, predicted.stderr)
        self.assertIn("Predicted intent:", predicted.stdout)
        self.assertIn("local CPU", predicted.stdout)
        canonical_run = output / selection["run"]
        alias_run = output / "runs" / f"{selection['configuration_index']:02d}"
        self.assertNotEqual(alias_run.resolve(), canonical_run.resolve())
        shutil.copytree(canonical_run, alias_run)
        (alias_run / "best-model.pt").write_bytes(b"unvalidated alias weights")
        alias_prediction = run_cli(
            "banking77.predict", "--model", alias_run,
            "--message", "my separate refund demonstration", cwd=self.workspace)
        self.assertEqual(alias_prediction.returncode, 0, alias_prediction.stderr)
        self.assertEqual(alias_prediction.stdout, predicted.stdout)

    def test_failed_submission_is_reported_and_does_not_prevent_other_independent_jobs(self):
        self.setup_launch()
        (self.workspace / "service").mkdir()
        (self.workspace / "service/control.json").write_text('{"fail_submit": [0]}')
        result = self.launch("--azure-device", "cpu", "--instance-type", "Standard_D2s_v3")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("synthetic submission quota error", result.stderr)
        manifest = json.loads((self.workspace / "comparison/launch.json").read_text())
        self.assertEqual(len(manifest["jobs"]), 2)
        self.assertEqual(manifest["jobs"][0]["status"], "SubmissionFailed")
        self.assertIn("quota error", manifest["jobs"][0]["error"])
        self.assertEqual(manifest["jobs"][1]["status"], "Queued")
        self.assertFalse((self.workspace / "comparison/selection.json").exists())

    def test_gpu_choice_is_explicit_and_unavailable_cuda_cannot_fall_back(self):
        self.setup_launch()
        for options, message in (
            ((), "--azure-device"),
            (("--azure-device", "gpu"), "--instance-type"),
        ):
            invalid = self.launch(*options)
            self.assertEqual(invalid.returncode, 1, invalid.stdout)
            self.assertIn(message, invalid.stderr)
            self.assertFalse((self.workspace / "comparison").exists())
        invalid = self.cli("launch", "--target", "local-cpu", "--configurations",
                           "configurations.json", "--output", "local-invalid",
                           "--azure-device", "gpu")
        self.assertEqual(invalid.returncode, 1, invalid.stdout)
        self.assertIn("Azure options require --target azure", invalid.stderr)
        self.assertFalse((self.workspace / "local-invalid").exists())
        submitted = self.launch("--azure-device", "gpu",
                                "--instance-type", "Standard_NC4as_T4_v3")
        self.assertEqual(submitted.returncode, 0, submitted.stderr)
        manifest = json.loads((self.workspace / "comparison/launch.json").read_text())
        self.assertEqual(manifest["device"], "cuda")
        self.assertIn("torch==2.10.0+cu126",
                      manifest["training_environment"]["conda"]["dependencies"][-1]["pip"])
        name = manifest["jobs"][0]["name"]
        # This deliberately runs the captured GPU worker on a CPU-only test host.
        failed = subprocess.run(
            [sys.executable, "-m", "banking77.azure_worker", "--index", "0",
             "--output", str(self.workspace / "gpu-output")],
            cwd=self.workspace / "service" / name / "code",
            capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        self.assertEqual(failed.returncode, 1, failed.stdout)
        self.assertIn("CUDA is unavailable; no CPU fallback", failed.stderr)
        self.assertFalse((self.workspace / "gpu-output/best-model.pt").exists())

    def test_collection_rejects_mixed_results_but_collects_other_completed_jobs_despite_failure(self):
        self.setup_launch()
        path = self.workspace / "configurations.json"
        settings = json.loads(path.read_text())
        settings.append({"embedding_size": 6, "hidden_size": 6, "max_epochs": 1})
        path.write_text(json.dumps(settings))
        launched = self.launch("--azure-device", "cpu", "--instance-type", "Standard_D2s_v3")
        self.assertEqual(launched.returncode, 0, launched.stderr)
        output = self.workspace / "comparison"
        manifest = json.loads((output / "launch.json").read_text())
        names = [job["name"] for job in manifest["jobs"]]
        completed = subprocess.run(
            [sys.executable, "-c", "from fake_azure_service import FakeAzure\n"
             f"service = FakeAzure('service')\n"
             f"service.complete({names[0]!r})\nservice.complete({names[1]!r})"],
            cwd=self.workspace, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        record_path = self.workspace / "service" / names[0] / "results/run.json"
        record = json.loads(record_path.read_text())
        record["provenance"]["launch_id"] = "another-comparison"
        record_path.write_text(json.dumps(record))
        failed_path = self.workspace / "service" / names[2] / "info.json"
        failed = json.loads(failed_path.read_text())
        failed.update(status="Failed", error="synthetic GPU/CPU job error")
        failed_path.write_text(json.dumps(failed))
        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 1, collected.stdout)
        self.assertIn("mismatched launch provenance", collected.stderr)
        self.assertIn("synthetic GPU/CPU job error", collected.stdout + collected.stderr)
        self.assertIn("1/3", collected.stdout)
        self.assertFalse((output / "runs/0").exists())
        self.assertTrue((output / "runs/1/best-model.pt").is_file())
        stored = json.loads((output / "launch.json").read_text())
        self.assertIsNone(stored["results"][0])
        self.assertIn("mismatched", stored["jobs"][0]["collection_error"])
        blocked = self.cli("select", "--launch", "comparison")
        self.assertEqual(blocked.returncode, 1, blocked.stdout)
        self.assertFalse((output / "selection.json").exists())

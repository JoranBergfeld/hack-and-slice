import json
import subprocess
import sys
import unittest

import torch

import test_azure_checkpoint as azure_tests


# Controlled participant work: real training, but scripted validation scores
# keyed on Adam's restored step count so the best epoch (2) differs from the
# latest completed epoch.
SCRIPTED_VALIDATION = '''
_scripted_accuracies = [0.5, 0.75, 0.25, 0.25, 0.25, 0.25]
_completed_epochs = [0]
_documented_train_epoch = train_epoch
def train_epoch(model, loader, optimizer, device):
    loss = _documented_train_epoch(model, loader, optimizer, device)
    steps = optimizer.state[next(iter(model.parameters()))]["step"]
    _completed_epochs[0] = int(steps) // len(loader)
    return loss
_documented_validation_accuracy = validation_accuracy
def validation_accuracy(model, loader, device):
    _documented_validation_accuracy(model, loader, device)
    return _scripted_accuracies[_completed_epochs[0] - 1]
'''
CONFIGURATION = {"embedding_size": 4, "hidden_size": 4, "batch_size": 8,
                 "max_epochs": 6, "patience": 3}


class RecoveryCheckpointTests(unittest.TestCase):
    setUp = azure_tests.AzureCheckpointTests.setUp
    install_documented_solution = azure_tests.AzureCheckpointTests.install_documented_solution
    prepare_synthetic_data = azure_tests.AzureCheckpointTests.prepare_synthetic_data
    setup_launch = azure_tests.AzureCheckpointTests.setup_launch
    cli = azure_tests.AzureCheckpointTests.cli
    launch = azure_tests.AzureCheckpointTests.launch

    def setup_recovery_launch(self, *options):
        self.setup_launch()
        with (self.workspace / "banking77/training.py").open("a") as module:
            module.write(SCRIPTED_VALIDATION)
        # Two identical configurations: the second is an uninterrupted reference.
        (self.workspace / "configurations.json").write_text(
            json.dumps([CONFIGURATION, CONFIGURATION]))
        launched = self.launch(*(options or ("--azure-device", "cpu",
                                             "--instance-type", "Standard_D2s_v3")))
        self.assertEqual(launched.returncode, 0, launched.stderr)
        self.output = self.workspace / "comparison"
        manifest = json.loads((self.output / "launch.json").read_text())
        return manifest, [job["name"] for job in manifest["jobs"]]

    def service(self, call):
        result = subprocess.run(
            [sys.executable, "-c", "from fake_azure_service import FakeAzure\n"
             f"service = FakeAzure('service')\n{call}"],
            cwd=self.workspace, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def break_working_copy(self):
        module = self.workspace / "banking77/training.py"
        participant_code = module.read_text()
        module.write_text("raise RuntimeError('later participant edits must not run')\n")
        (self.workspace / "configurations.json").write_text("[]")
        return participant_code

    def test_interrupted_job_resumes_explicitly_from_latest_snapshot_like_an_uninterrupted_run(self):
        manifest, (interrupted, reference) = self.setup_recovery_launch()
        self.service(f"service.interrupt({interrupted!r}, after_epochs=3)\n"
                     f"service.complete({reference!r})")
        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 1, collected.stdout)
        self.assertIn("synthetic node preemption during epoch 4", collected.stdout)
        self.assertIn("1/2", collected.stdout)
        reference_model = (self.output / "runs/1/best-model.pt").read_bytes()
        participant_code = self.break_working_copy()

        recovered = self.cli("recover", "--launch", "comparison",
                             "--job", interrupted, "--resume")
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        self.assertIn("after epoch 3", recovered.stdout)
        self.assertIn("best validation 0.7500 at epoch 2", recovered.stdout)
        stored = json.loads((self.output / "launch.json").read_text())
        self.assertEqual(stored["launch_id"], manifest["launch_id"])
        self.assertEqual(stored["jobs"][1]["name"], reference)
        resumed = stored["jobs"][0]["name"]
        self.assertNotEqual(resumed, interrupted)
        self.assertIn(resumed, recovered.stdout)
        self.assertIsNotNone(stored["results"][1])
        self.assertEqual((self.output / "runs/1/best-model.pt").read_bytes(), reference_model)

        self.service(f"service.complete({resumed!r})")
        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 0, collected.stderr)
        self.assertIn("2/2", collected.stdout)
        runs = [json.loads((self.output / f"runs/{index}/run.json").read_text())
                for index in range(2)]
        self.assertEqual([entry["validation_accuracy"] for entry in runs[0]["history"]],
                         [0.5, 0.75, 0.25, 0.25, 0.25])
        # Restored model, optimizer, data-order and dropout randomness reproduce
        # the uninterrupted CPU run exactly.
        self.assertEqual(runs[0]["history"], runs[1]["history"])
        self.assertEqual(runs[0]["best_epoch"], 2)
        self.assertTrue(runs[0]["stopped_early"])
        self.assertEqual(runs[0]["provenance"]["launch_id"], manifest["launch_id"])
        self.assertEqual(runs[0]["recovery"]["mode"], "resume")
        self.assertEqual(runs[0]["recovery"]["failed_job"], interrupted)
        self.assertEqual(runs[0]["recovery"]["resumed_after_epoch"], 3)
        for name in ("best-model.pt", "latest-snapshot.pt"):
            states = [torch.load(self.output / f"runs/{index}" / name,
                                 weights_only=True) for index in range(2)]
            key = "state_dict" if name == "best-model.pt" else "model_state"
            for parameter, tensor in states[1][key].items():
                self.assertTrue(torch.equal(states[0][key][parameter], tensor), parameter)
        (self.workspace / "banking77/training.py").write_text(participant_code)
        selected = self.cli("select", "--launch", "comparison")
        self.assertEqual(selected.returncode, 0, selected.stderr)

    def test_job_interrupted_before_any_snapshot_restarts_only_when_asked(self):
        manifest, (interrupted, reference) = self.setup_recovery_launch()
        self.service(f"service.interrupt({interrupted!r}, after_epochs=0)\n"
                     f"service.complete({reference!r})")
        self.assertEqual(self.cli("collect", "--launch", "comparison").returncode, 1)
        before = (self.output / "launch.json").read_text()

        completed = self.cli("recover", "--launch", "comparison", "--job", reference,
                             "--restart")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("already has a collected result", completed.stderr)
        resumed = self.cli("recover", "--launch", "comparison", "--job", interrupted,
                           "--resume")
        self.assertEqual(resumed.returncode, 1)
        self.assertIn("no usable latest resumable snapshot", resumed.stderr)
        self.assertIn("--restart", resumed.stderr)
        self.assertEqual((self.output / "launch.json").read_text(), before)

        restarted = self.cli("recover", "--launch", "comparison", "--job", interrupted,
                             "--restart")
        self.assertEqual(restarted.returncode, 0, restarted.stderr)
        stored = json.loads((self.output / "launch.json").read_text())
        self.assertEqual(stored["launch_id"], manifest["launch_id"])
        self.service(f"service.complete({stored['jobs'][0]['name']!r})")
        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 0, collected.stderr)
        runs = [json.loads((self.output / f"runs/{index}/run.json").read_text())
                for index in range(2)]
        self.assertEqual(runs[0]["recovery"]["mode"], "restart")
        self.assertIsNone(runs[0]["recovery"]["resumed_after_epoch"])
        self.assertIsNone(runs[0]["resumed"])
        self.assertEqual(runs[0]["history"], runs[1]["history"])

    def test_failed_resume_submission_can_retry_from_the_retained_snapshot(self):
        _, (interrupted, reference) = self.setup_recovery_launch()
        self.service(f"service.interrupt({interrupted!r}, after_epochs=3)\n"
                     f"service.complete({reference!r})")
        self.assertEqual(self.cli("collect", "--launch", "comparison").returncode, 1)

        control = self.workspace / "service/control.json"
        control.write_text('{"fail_submit": [0]}')
        failed = self.cli("recover", "--launch", "comparison",
                          "--job", interrupted, "--resume")
        self.assertEqual(failed.returncode, 1)
        self.assertIn("synthetic submission quota error", failed.stderr)
        stored = json.loads((self.output / "launch.json").read_text())
        failed_submission = stored["jobs"][0]["name"]
        self.assertEqual(stored["jobs"][0]["status"], "SubmissionFailed")
        retained = self.output / "recoveries/0-r1/latest-snapshot.pt"
        self.assertTrue(retained.is_file())

        control.unlink()
        retried = self.cli("recover", "--launch", "comparison",
                           "--job", failed_submission, "--resume")
        self.assertEqual(retried.returncode, 0, retried.stderr)
        stored = json.loads((self.output / "launch.json").read_text())
        retry = stored["jobs"][0]["name"]
        self.assertEqual(stored["recoveries"][1]["resumed_after_epoch"], 3)
        self.assertEqual(stored["recoveries"][1]["failed_job"], failed_submission)

        self.service(f"service.complete({retry!r})")
        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 0, collected.stderr)
        self.assertEqual(json.loads(
            (self.output / "runs/0/run.json").read_text()
        )["recovery"]["mode"], "resume")

    def test_unusable_remote_snapshot_is_rejected_before_submitting_paid_job(self):
        _, (interrupted, _) = self.setup_recovery_launch()
        self.service(f"service.interrupt({interrupted!r}, after_epochs=3)")
        self.assertEqual(self.cli("collect", "--launch", "comparison").returncode, 1)
        remote = self.workspace / "service" / interrupted / "results/latest-snapshot.pt"
        import torch
        snapshot = torch.load(remote, map_location="cpu", weights_only=True)
        snapshot["model_state"].pop(next(iter(snapshot["model_state"])))
        torch.save(snapshot, remote)

        rejected = self.cli("recover", "--launch", "comparison",
                            "--job", interrupted, "--resume")
        self.assertEqual(rejected.returncode, 1, rejected.stdout)
        self.assertIn("latest model state does not match", rejected.stderr)
        manifest = json.loads((self.output / "launch.json").read_text())
        self.assertEqual(manifest["jobs"][0]["name"], interrupted)
        self.assertNotIn("recoveries", manifest)
        self.assertFalse(any(path.name.endswith("-r1")
                             for path in (self.workspace / "service").iterdir()))

    def test_resume_reuses_a_verified_recovery_bundle_after_manifest_update_interruption(self):
        _, (interrupted, _) = self.setup_recovery_launch()
        self.service(f"service.interrupt({interrupted!r}, after_epochs=3)")
        self.assertEqual(self.cli("collect", "--launch", "comparison").returncode, 1)
        control = self.workspace / "service/control.json"
        control.write_text('{"fail_submit": [0]}')

        first = self.cli("recover", "--launch", "comparison",
                         "--job", interrupted, "--resume")
        self.assertEqual(first.returncode, 1)
        bundle = self.output / "recoveries/0-r1"
        self.assertTrue((bundle / "banking77/model.py").is_file())
        self.assertFalse((bundle / "submission").exists())

        manifest_path = self.output / "launch.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["jobs"][0] = {"name": interrupted, "status": "Failed",
                               "error": "synthetic node preemption"}
        manifest.pop("recoveries")
        manifest_path.write_text(json.dumps(manifest))
        control.unlink()

        retried = self.cli("recover", "--launch", "comparison",
                           "--job", interrupted, "--resume")
        self.assertEqual(retried.returncode, 0, retried.stderr)
        stored = json.loads(manifest_path.read_text())
        self.assertEqual(len(stored["recoveries"]), 1)
        self.assertEqual(stored["recoveries"][0]["job_name"], stored["jobs"][0]["name"])

    def test_snapshot_with_changed_optimizer_options_is_rejected_before_submission(self):
        _, (interrupted, _) = self.setup_recovery_launch()
        self.service(f"service.interrupt({interrupted!r}, after_epochs=3)")
        self.assertEqual(self.cli("collect", "--launch", "comparison").returncode, 1)
        remote = self.workspace / "service" / interrupted / "results/latest-snapshot.pt"
        snapshot = torch.load(remote, map_location="cpu", weights_only=True)
        snapshot["optimizer_state"]["param_groups"][0]["lr"] *= 2
        torch.save(snapshot, remote)

        rejected = self.cli("recover", "--launch", "comparison",
                            "--job", interrupted, "--resume")
        self.assertEqual(rejected.returncode, 1, rejected.stdout)
        self.assertIn("optimizer options do not match", rejected.stderr)
        manifest = json.loads((self.output / "launch.json").read_text())
        self.assertEqual(manifest["jobs"][0]["name"], interrupted)
        self.assertNotIn("recoveries", manifest)

    def test_failed_recovery_without_remote_progress_falls_back_to_retained_snapshot(self):
        _, (interrupted, _) = self.setup_recovery_launch()
        self.service(f"service.interrupt({interrupted!r}, after_epochs=3)")
        self.assertEqual(self.cli("collect", "--launch", "comparison").returncode, 1)
        first = self.cli("recover", "--launch", "comparison",
                         "--job", interrupted, "--resume")
        self.assertEqual(first.returncode, 0, first.stderr)
        first_retry = json.loads(
            (self.output / "launch.json").read_text()
        )["jobs"][0]["name"]
        self.service(f"service.fail({first_retry!r}, 'failed before worker startup')")

        retried = self.cli("recover", "--launch", "comparison",
                           "--job", first_retry, "--resume")
        self.assertEqual(retried.returncode, 0, retried.stderr)
        self.assertIn("Using retained snapshot", retried.stdout)
        manifest = json.loads((self.output / "launch.json").read_text())
        self.assertEqual(manifest["recoveries"][1]["resumed_after_epoch"], 3)
        self.assertEqual(manifest["recoveries"][1]["failed_job"], first_retry)
        self.assertEqual(manifest["recoveries"][1]["snapshot_sha256"],
                         manifest["recoveries"][0]["snapshot_sha256"])

    def test_gpu_job_resumes_on_cpu_only_when_explicitly_chosen(self):
        manifest, (interrupted, untouched) = self.setup_recovery_launch(
            "--azure-device", "gpu", "--instance-type", "Standard_NC4as_T4_v3")
        self.service(f"service.interrupt({interrupted!r}, after_epochs=3, "
                     "emulate_gpu_on_cpu=True)")
        self.assertEqual(self.cli("collect", "--launch", "comparison").returncode, 1)

        same_device = self.cli("recover", "--launch", "comparison", "--job", interrupted,
                               "--resume")
        self.assertEqual(same_device.returncode, 0, same_device.stderr)
        stored = json.loads((self.output / "launch.json").read_text())
        gpu_retry = stored["jobs"][0]["name"]
        self.assertEqual(stored["recoveries"][0]["compute"]["device"], "gpu")
        # No CPU fallback: the GPU retry fails on a node without CUDA.
        result = subprocess.run(
            [sys.executable, "-c", "from fake_azure_service import FakeAzure\n"
             f"FakeAzure('service').complete({gpu_retry!r})"],
            cwd=self.workspace, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no CPU fallback", result.stderr)
        self.service(f"service.fail({gpu_retry!r}, 'CUDA unavailable')")

        partial = self.cli("recover", "--launch", "comparison", "--job", gpu_retry,
                           "--resume", "--azure-device", "cpu")
        self.assertEqual(partial.returncode, 1)
        self.assertIn("--instance-type", partial.stderr)
        on_cpu = self.cli("recover", "--launch", "comparison", "--job", gpu_retry,
                          "--resume", "--azure-device", "cpu",
                          "--instance-type", "Standard_D2s_v3")
        self.assertEqual(on_cpu.returncode, 0, on_cpu.stderr)
        self.assertIn("after epoch 3 on cuda", on_cpu.stdout)
        self.assertIn("not guaranteed bit-for-bit identical", on_cpu.stdout)
        stored = json.loads((self.output / "launch.json").read_text())
        self.assertEqual(stored["launch_id"], manifest["launch_id"])
        self.assertEqual(stored["jobs"][1]["name"], untouched)
        self.assertEqual(stored["device"], "cuda")
        self.service(f"service.complete({stored['jobs'][0]['name']!r})")
        collected = self.cli("collect", "--launch", "comparison")
        self.assertEqual(collected.returncode, 0, collected.stderr)
        self.assertIn("1/2", collected.stdout)
        record = json.loads((self.output / "runs/0/run.json").read_text())
        self.assertEqual(record["device"], "cpu")
        self.assertEqual(record["compute"]["device"], "cpu")
        self.assertTrue(record["software"]["torch"].endswith("+cpu"))
        self.assertEqual(record["recovery"]["from_device"], "cuda")
        self.assertEqual(record["recovery"]["failed_job"], gpu_retry)
        self.assertEqual([entry["device"] for entry in record["history"]],
                         ["cuda"] * 3 + ["cpu"] * 2)
        self.assertEqual(record["best_epoch"], 2)

    def test_completed_rejected_output_can_be_explicitly_restarted_or_resumed(self):
        for mode in ("restart", "resume"):
            with self.subTest(mode=mode):
                if mode == "resume":
                    self.setUp()
                manifest, (rejected, reference) = self.setup_recovery_launch()
                running = self.cli("recover", "--launch", "comparison",
                                   "--job", rejected, "--restart")
                self.assertEqual(running.returncode, 1, running.stdout)
                self.assertIn("Queued", running.stderr)
                self.service(f"service.complete({rejected!r})\n"
                             f"service.complete({reference!r})")
                uncollected = self.cli("recover", "--launch", "comparison",
                                       "--job", rejected, "--restart")
                self.assertEqual(uncollected.returncode, 1, uncollected.stdout)
                self.assertIn("run collect first", uncollected.stderr)
                results = self.workspace / "service" / rejected / "results"
                if mode == "restart":
                    (results / "best-model.pt").unlink()
                else:
                    record_path = results / "run.json"
                    record = json.loads(record_path.read_text())
                    record["software"]["torch"] = "unsupported-version"
                    record_path.write_text(json.dumps(record))
                collected = self.cli("collect", "--launch", "comparison")
                self.assertEqual(collected.returncode, 1, collected.stdout)
                self.assertIn("Collection error:", collected.stdout)
                self.assertIn("1/2", collected.stdout)
                # Refreshing status must not erase the reason recovery is allowed.
                status = self.cli("status", "--launch", "comparison")
                self.assertIn("Completed", status.stdout)
                self.assertIn("Collection error:", status.stdout)
                if mode == "resume":
                    info_path = self.workspace / "service" / rejected / "info.json"
                    info = json.loads(info_path.read_text())
                    info["status"] = "Running"
                    info_path.write_text(json.dumps(info))
                    running = self.cli("recover", "--launch", "comparison",
                                       "--job", rejected, "--resume")
                    self.assertEqual(running.returncode, 1, running.stdout)
                    self.assertIn("Running", running.stderr)
                    info["status"] = "Completed"
                    info_path.write_text(json.dumps(info))
                before = (self.output / "runs/1/best-model.pt").read_bytes()
                self.break_working_copy()
                recovered = self.cli("recover", "--launch", "comparison",
                                     "--job", rejected, "--" + mode)
                self.assertEqual(recovered.returncode, 0, recovered.stderr)
                stored = json.loads((self.output / "launch.json").read_text())
                self.assertEqual(stored["launch_id"], manifest["launch_id"])
                self.assertEqual(stored["jobs"][1]["name"], reference)
                retry = stored["jobs"][0]["name"]
                self.service(f"service.complete({retry!r})")
                collected = self.cli("collect", "--launch", "comparison")
                self.assertEqual(collected.returncode, 0, collected.stderr)
                self.assertIn("2/2", collected.stdout)
                self.assertEqual((self.output / "runs/1/best-model.pt").read_bytes(), before)
                record = json.loads((self.output / "runs/0/run.json").read_text())
                self.assertEqual(record["recovery"]["mode"], mode)
                blocked = self.cli("recover", "--launch", "comparison",
                                   "--job", retry, "--restart")
                self.assertEqual(blocked.returncode, 1)
                self.assertIn("already has a collected result", blocked.stderr)


if __name__ == "__main__":
    unittest.main()

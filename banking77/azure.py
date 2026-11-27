"""Azure ML service boundary and independent captured comparison submissions."""
import copy
import json
from pathlib import Path
import pickle
import re
import shutil
import tempfile
from typing import Protocol

from banking77.compare import (
    RESULT_FILES, capture, validate_capture_inputs, validate_manifest_results,
    validate_results, write_chart,
)
from banking77.artifacts import file_sha256, launch_lock, write_json
from banking77.azure_settings import job_name, torch_build, torch_device
from banking77.captured import captured_modules
from banking77.runner import LATEST_SNAPSHOT

RECOVERABLE = ("Failed", "Canceled", "SubmissionFailed")


class AzureJobNotFoundError(RuntimeError):
    """Azure confirmed that a submitted job name does not exist."""


class AzureService(Protocol):
    """Only external job operations; offline tests substitute this boundary."""

    def submit(self, bundle: Path, name: str, index: int, compute: dict) -> dict: ...
    def inspect(self, name: str) -> dict: ...
    def download(self, name: str, destination: Path) -> Path: ...


def _show(job: dict) -> None:
    print(f"{job['name']}: {job['status']}", flush=True)
    print(f"  Azure ML logs/progress: {job.get('studio_url')}", flush=True)
    print(f"  Output: {job.get('output_uri')}", flush=True)
    if job.get("error"):
        print(f"  Error: {job['error']}", flush=True)
    if job.get("collection_error"):
        print(f"  Collection error: {job['collection_error']}; "
              "retry collect or explicitly recover with --restart/--resume.", flush=True)


def _verified_tree(directory: Path, expected: dict) -> bool:
    if not directory.is_dir():
        return False
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        return False
    actual = {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in paths if path.is_file()
    }
    return actual == expected


def _recovery_bundle_matches(
    bundle: Path, manifest: dict, entry: dict, environment: dict
) -> bool:
    try:
        bundled_manifest = json.loads((bundle / "launch.json").read_text(encoding="utf-8"))
        for key in ("schema_version", "launch_id", "target", "configurations", "seed",
                    "device", "examples", "software", "environment", "source", "prepared",
                    "azure", "training_environment"):
            if bundled_manifest.get(key) != manifest.get(key):
                return False
        if json.loads((bundle / "recovery.json").read_text(encoding="utf-8")) != entry:
            return False
        if json.loads((bundle / "environment.json").read_text(encoding="utf-8")) != environment:
            return False
        source_files = {}
        metadata = {"launch.json", "recovery.json", "environment.json", LATEST_SNAPSHOT}
        for path in bundle.rglob("*"):
            if path.is_symlink():
                return False
            if not path.is_file():
                continue
            relative = path.relative_to(bundle).as_posix()
            if relative.startswith("prepared/") or relative in metadata:
                continue
            source_files[relative] = file_sha256(path)
        if source_files != manifest["source"]["sha256"]:
            return False
        if not _verified_tree(bundle / "prepared", manifest["prepared"]["sha256"]):
            return False
        snapshot = bundle / LATEST_SNAPSHOT
        if entry["snapshot_sha256"] is None:
            return not snapshot.exists()
        return snapshot.is_file() and file_sha256(snapshot) == entry["snapshot_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _training_environment(output: Path, software: dict, azure_device: str) -> dict:
    """Image plus conda definition with the torch build for the chosen device."""
    torch_version = software["torch"].split("+")[0]
    build = torch_build(torch_device(azure_device))
    requirements = output / "source" / "requirements.txt"
    extra_packages = []
    if requirements.is_file():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-", ".")) or " @ " in line:
                raise ValueError("Azure requirements must be package specifiers, not local files/options")
            if not re.match(r"^(torch|scikit-learn)\s*(==|>=|<=|~=|!=|>|<|$)", line):
                extra_packages.append(line)
            elif re.match(r"^torch\s*==", line):
                torch_version = re.sub(r"^torch\s*==\s*", "", line)
    return {
        "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        "conda": {"name": "banking77", "channels": ["conda-forge"],
                  "dependencies": [
                      f"python={software['python']}", "pip",
                      {"pip": [f"--extra-index-url https://download.pytorch.org/whl/{build}",
                               f"torch=={torch_version}+{build}",
                               f"scikit-learn=={software['scikit-learn']}", *extra_packages]},
                  ]},
    }


def launch_azure(arguments, *, service: AzureService | None = None) -> None:
    for option in ("azure_device", "instance_type", "subscription", "resource_group", "workspace"):
        if not getattr(arguments, option):
            raise ValueError(f"--{option.replace('_', '-')} is required for Azure; no fallback")
    output = arguments.output.resolve()
    validate_capture_inputs(arguments.configurations, arguments.prepared, arguments.examples)
    output.mkdir(parents=True, exist_ok=False)
    with launch_lock(output):
        manifest = capture(
            arguments.configurations, arguments.prepared, output,
            examples=arguments.examples, create_output=False,
        )
        _submit_captured_launch(arguments, service, output, manifest)


def _submit_captured_launch(arguments, service, output: Path, manifest: dict) -> None:
    manifest.update(
        target="azure", device=torch_device(arguments.azure_device),
        status="submitting", results=[None] * len(manifest["configurations"]), jobs=[],
        azure={"subscription": arguments.subscription, "resource_group": arguments.resource_group,
               "workspace": arguments.workspace, "device": arguments.azure_device,
               "instance_type": arguments.instance_type, "tier": "Standard"},
    )
    requirements = output / "source" / "requirements.txt"
    if requirements.is_file():
        pinned_torch = re.search(
            r"(?m)^\s*torch\s*==\s*([^\s#]+)", requirements.read_text(encoding="utf-8"))
        if pinned_torch:
            manifest["software"]["torch"] = pinned_torch.group(1)
    # Capture the exact environment definition alongside the submitted code.
    manifest["training_environment"] = _training_environment(
        output, manifest["software"], arguments.azure_device)
    write_json(output / "launch.json", manifest)
    bundle = output / "submission"
    shutil.copytree(output / "source", bundle)
    shutil.copytree(output / "prepared", bundle / "prepared")
    write_json(bundle / "launch.json", manifest)
    write_json(bundle / "environment.json", manifest["training_environment"]["conda"])
    print(f"Target: Azure ML {arguments.azure_device.upper()}, "
          f"{arguments.instance_type}, Standard tier", flush=True)
    try:
        service = service or MLService(manifest["azure"], manifest["training_environment"])
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        manifest.update(status="setup_failed", error=str(error))
        write_json(output / "launch.json", manifest)
        raise RuntimeError(f"Azure setup failed: {error}; install requirements-azure.txt "
                           "and authenticate with az login") from error
    failures = []
    for index in range(len(manifest["configurations"])):
        name = job_name(manifest["launch_id"], index)
        job = {"name": name, "status": "Submitting", "error": None}
        manifest["jobs"].append(job)
        write_json(output / "launch.json", manifest)
        try:
            job = service.submit(bundle, name, index, manifest["azure"])
        except (RuntimeError, OSError, ValueError) as error:
            job.update(status="SubmissionFailed", error=str(error))
            failures.append(str(error))
        manifest["jobs"][index] = job
        write_json(output / "launch.json", manifest)
        _show(job)
    manifest["status"] = "submission_failed" if failures else "submitted"
    write_json(output / "launch.json", manifest)
    if failures:
        raise RuntimeError("; ".join(failures) + "; other submitted jobs remain available")


def inspect_azure(output: Path, *, collect: bool, service: AzureService | None = None) -> None:
    output = output.resolve()
    with launch_lock(output):
        _inspect_azure_locked(output, collect=collect, service=service)


def _inspect_azure_locked(
    output: Path, *, collect: bool, service: AzureService | None = None
) -> None:
    manifest = json.loads((output / "launch.json").read_text(encoding="utf-8"))
    if manifest.get("target") != "azure":
        raise ValueError("status/collect requires an Azure launch")
    validate_results(output, allow_partial=True)
    service = service or MLService(manifest["azure"], manifest["training_environment"])
    print(f"Target: Azure ML {manifest['azure']['device'].upper()}, "
          f"{manifest['azure']['instance_type']}, Standard tier", flush=True)
    failures = [manifest["error"]] if manifest["status"] == "setup_failed" else []
    for index, previous in enumerate(manifest["jobs"]):
        try:
            job = service.inspect(previous["name"])
            if job["name"] != previous["name"]:
                raise ValueError("Azure status returned a different job identifier")
            if previous.get("collection_error"):
                job["collection_error"] = previous["collection_error"]
        except (RuntimeError, OSError, ValueError) as error:
            job = {**previous, "error": str(error)}
            failures.append(str(error))
        manifest["jobs"][index] = job
        _show(job)
        if job["status"] in RECOVERABLE:
            failures.append(f"{job['name']}: {job.get('error') or job['status']}")
        if collect and job["status"] == "Completed" and manifest["results"][index] is None:
            try:
                with tempfile.TemporaryDirectory(prefix="collect-", dir=output) as temporary:
                    downloaded = service.download(job["name"], Path(temporary) / "download")
                    for name in RESULT_FILES:
                        if not (downloaded / name).is_file():
                            raise ValueError(f"{job['name']}: incomplete output: {name}")
                    record = json.loads((downloaded / "run.json").read_text(encoding="utf-8"))
                    candidate = copy.deepcopy(manifest)
                    candidate["results"][index] = {
                        "sha256": {name: file_sha256(downloaded / name) for name in RESULT_FILES},
                        "software": record["software"],
                    }
                    validate_manifest_results(output, candidate, allow_partial=True,
                                              pending_run=(index, downloaded))
                    run = output / "runs" / str(index)
                    run.parent.mkdir(exist_ok=True)
                    if run.exists():
                        validate_manifest_results(output, candidate, allow_partial=True,
                                                  pending_run=(index, run))
                    else:
                        downloaded.rename(run)
                    manifest["results"][index] = candidate["results"][index]
                    job.pop("collection_error", None)
                    progress = run / "progress.log"
                    if progress.is_file():
                        print(progress.read_text(encoding="utf-8"), flush=True)
            except (ValueError, OSError, RuntimeError, KeyError, TypeError, IndexError,
                    EOFError, pickle.UnpicklingError) as error:
                job["collection_error"] = str(error)
                failures.append(f"{job['name']}: {error}")
                print(f"  Collection error: {error}", flush=True)
        write_json(output / "launch.json", manifest)
    count = sum(result is not None for result in manifest["results"])
    if collect:
        if count == len(manifest["configurations"]):
            manifest["status"] = "completed"
            write_json(output / "launch.json", manifest)
            write_chart(output, validate_results(output))
        print(f"Collected {count}/{len(manifest['configurations'])} configurations. "
              "Selection requires every valid matching result.", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def _validate_recovery_snapshot(
    snapshot: Path, output: Path, manifest: dict, index: int,
    load_snapshot, configuration_type,
) -> dict:
    configuration = configuration_type(**manifest["configurations"][index])
    vocabulary = json.loads((output / "prepared" / "vocabulary.json").read_text(
        encoding="utf-8"))
    labels = json.loads((output / "prepared" / "labels.json").read_text(encoding="utf-8"))
    settings = configuration.model_settings(len(vocabulary), len(labels))
    with captured_modules(output / "source", ("data", "model", "training")) as captured:
        return load_snapshot(
            snapshot,
            configuration=configuration,
            model_settings=settings,
            identity={"launch_id": manifest["launch_id"], "configuration_index": index},
            prepared_sha256=manifest["prepared"]["sha256"],
            model_factory=lambda model_settings: captured["model"].IntentClassifier(
                **model_settings),
            optimizer_factory=lambda model, learning_rate: captured["training"].create_optimizer(
                model, learning_rate),
        )


def recover_azure(arguments, *, service: AzureService | None = None) -> None:
    output = arguments.launch.resolve()
    with launch_lock(output):
        _recover_azure_locked(arguments, output, service=service)


def _recover_azure_locked(
    arguments, output: Path, *, service: AzureService | None = None
) -> None:
    """Recover one failed job or completed job whose collected output was rejected."""
    from banking77.compare import expected_attempt
    from banking77.runner import LATEST_SNAPSHOT, TrainingConfiguration, load_snapshot
    manifest = json.loads((output / "launch.json").read_text(encoding="utf-8"))
    if manifest.get("target") != "azure":
        raise ValueError("recover requires an Azure launch")
    validate_results(output, allow_partial=True)
    names = [job["name"] for job in manifest["jobs"]]
    if arguments.job not in names:
        raise ValueError(f"{arguments.job} is not the current job of any configuration "
                         "in this launch")
    index = names.index(arguments.job)
    if manifest["results"][index] is not None:
        raise ValueError(f"{arguments.job} already has a collected result; nothing to recover")
    if bool(arguments.azure_device) != bool(arguments.instance_type):
        raise ValueError("--azure-device and --instance-type must be given together; no fallback")
    service = service or MLService(manifest["azure"], manifest["training_environment"])
    job = manifest["jobs"][index]
    collection_error = job.get("collection_error")
    confirmed_not_found = False
    if job["status"] != "SubmissionFailed":
        job = service.inspect(arguments.job)
        if job["name"] != arguments.job:
            raise ValueError("Azure status returned a different job identifier")
        manifest["jobs"][index] = job
    else:
        try:
            job = service.inspect(arguments.job)
        except AzureJobNotFoundError:
            confirmed_not_found = True
        else:
            if job["name"] != arguments.job:
                raise ValueError("Azure status returned a different job identifier")
            manifest["jobs"][index] = job
    if (job["status"] not in RECOVERABLE
            and not (confirmed_not_found and
                     manifest["jobs"][index]["status"] == "SubmissionFailed")
            and not (job["status"] == "Completed" and collection_error)):
        raise ValueError(f"{arguments.job} is {job['status']}; only Failed or Canceled "
                         "jobs, failed submissions, or Completed jobs whose output was "
                         "rejected by collect can be recovered; run collect first")
    previous = expected_attempt(manifest, index)
    if arguments.azure_device:
        device = torch_device(arguments.azure_device)
        compute = {**manifest["azure"], "device": arguments.azure_device,
                   "instance_type": arguments.instance_type}
        environment = {**manifest["training_environment"],
                       "conda": _training_environment(output, manifest["software"],
                                                      arguments.azure_device)["conda"]}
    else:
        device, compute = previous["device"], previous["compute"]
        environment = previous["training_environment"]
    attempt = 1 + sum(entry["configuration_index"] == index
                      for entry in manifest.get("recoveries", []))
    name = job_name(manifest["launch_id"], index, attempt=attempt)
    bundle = output / "recoveries" / f"{index}-r{attempt}"
    with tempfile.TemporaryDirectory(prefix="recover-", dir=output) as temporary:
        snapshot, entry = None, {
            "configuration_index": index, "mode": "resume" if arguments.resume else "restart",
            "attempt": attempt, "failed_job": arguments.job, "job_name": name,
            "device": device, "compute": compute, "training_environment": environment,
            "from_device": previous["device"], "resumed_after_epoch": None,
            "snapshot_sha256": None,
        }
        if arguments.resume:
            prior_recovery = previous["recovery"]
            retained_snapshot = None
            retained_error = None
            if prior_recovery and prior_recovery.get("snapshot_sha256"):
                retained_snapshot = (
                    output / "recoveries" / f"{index}-r{prior_recovery['attempt']}"
                    / LATEST_SNAPSHOT
                )
            try:
                if job["status"] != "SubmissionFailed" and not confirmed_not_found:
                    downloaded = service.download(
                        arguments.job, Path(temporary) / "download")
                    snapshot = downloaded / LATEST_SNAPSHOT
                    state = _validate_recovery_snapshot(
                        snapshot, output, manifest, index, load_snapshot,
                        TrainingConfiguration)
                else:
                    raise ValueError("submission did not produce remote progress")
            except (OSError, RuntimeError, ValueError, KeyError, EOFError,
                    pickle.UnpicklingError) as error:
                retained_error = error
                state = None
            if state is None and retained_snapshot is not None:
                try:
                    if file_sha256(retained_snapshot) != prior_recovery["snapshot_sha256"]:
                        raise ValueError("retained snapshot hash does not match")
                    snapshot = retained_snapshot
                    state = _validate_recovery_snapshot(
                        snapshot, output, manifest, index, load_snapshot,
                        TrainingConfiguration)
                    print(f"Using retained snapshot because {arguments.job} has no "
                          f"usable remote progress ({retained_error}).", flush=True)
                except (OSError, RuntimeError, ValueError, KeyError, EOFError,
                        pickle.UnpicklingError) as error:
                    raise ValueError(
                        f"{arguments.job} has no usable remote or retained latest "
                        f"resumable snapshot ({error}); use --restart to train it again "
                        "from epoch 1"
                    ) from error
            if state is None:
                raise ValueError(
                    f"{arguments.job} has no usable latest resumable snapshot "
                    f"({retained_error}); use --restart to train it again from epoch 1"
                ) from retained_error
            best = state["best"]
            entry.update(resumed_after_epoch=state["completed_epoch"],
                         from_device=state["device"], snapshot_sha256=file_sha256(snapshot))
            print(f"Latest resumable snapshot: after epoch {state['completed_epoch']} on "
                  f"{state['device']}; best validation {best['validation_accuracy']:.4f} "
                  f"at epoch {best['epoch']}.", flush=True)
            if state["device"] != device:
                print(f"Continuing on {device} after {state['device']}: valid, but not "
                      "guaranteed bit-for-bit identical to an uninterrupted run.", flush=True)
        bundle.parent.mkdir(exist_ok=True)
        if bundle.exists():
            if not _recovery_bundle_matches(bundle, manifest, entry, environment["conda"]):
                raise FileExistsError(
                    f"{bundle} already exists but does not match this recovery attempt"
                )
        else:
            with tempfile.TemporaryDirectory(
                prefix=f".{bundle.name}-", dir=bundle.parent
            ) as staging:
                staged_bundle = Path(staging) / "bundle"
                shutil.copytree(output / "source", staged_bundle)
                shutil.copytree(output / "prepared", staged_bundle / "prepared")
                write_json(staged_bundle / "launch.json", manifest)
                write_json(staged_bundle / "recovery.json", entry)
                write_json(staged_bundle / "environment.json", environment["conda"])
                if snapshot is not None:
                    shutil.copy2(snapshot, staged_bundle / LATEST_SNAPSHOT)
                staged_bundle.rename(bundle)
    manifest.setdefault("recoveries", []).append(entry)
    manifest["jobs"][index] = {"name": name, "status": "Submitting", "error": None}
    write_json(output / "launch.json", manifest)
    print(f"Target: Azure ML {compute['device'].upper()}, {compute['instance_type']}, "
          "Standard tier", flush=True)
    try:
        submitted = service.submit(bundle, name, index, compute)
    except (RuntimeError, OSError, ValueError) as error:
        submitted = {"name": name, "status": "SubmissionFailed", "error": str(error)}
    manifest["jobs"][index] = submitted
    write_json(output / "launch.json", manifest)
    print(f"Recovering configuration {index + 1} ({entry['mode']}) as {name}; "
          f"replaces {arguments.job}. Completed configurations are unchanged.", flush=True)
    _show(submitted)
    if submitted["status"] == "SubmissionFailed":
        raise RuntimeError(f"{name}: {submitted['error']}; recover again when resolved")


class MLService:
    """Real SDK v2 adapter; never creates/deletes a workspace or compute cluster."""

    def __init__(self, workspace: dict, environment: dict):
        from azure.ai.ml import MLClient
        from azure.identity import AzureCliCredential
        self.client = MLClient(AzureCliCredential(), workspace["subscription"],
                               workspace["resource_group"], workspace["workspace"])
        self.environment = environment
        self._checked_compute = set()

    def submit(self, bundle: Path, name: str, index: int, compute: dict) -> dict:
        from azure.ai.ml import command, Output
        from azure.ai.ml.entities import (
            Environment, JobResourceConfiguration, QueueSettings, UserIdentityConfiguration,
        )
        from azure.core.exceptions import AzureError
        choice = (compute["instance_type"], compute["device"])
        if choice not in self._checked_compute:
            try:
                size = next((size for size in self.client.compute.list_sizes()
                             if size.name.lower() == compute["instance_type"].lower()), None)
            except AzureError as error:
                raise RuntimeError(f"Azure compute/access check failed: {error}") from error
            if size is None:
                raise ValueError(f"Azure VM size {compute['instance_type']} is not available "
                                 "in this workspace region; no fallback")
            if size.gpus is None or (size.gpus > 0) != (compute["device"] == "gpu"):
                raise ValueError(f"Azure VM size {compute['instance_type']} does not match "
                                 f"the explicit {compute['device']} choice; no fallback")
            self._checked_compute.add(choice)
        manifest = json.loads((bundle / "launch.json").read_text(encoding="utf-8"))
        job = command(
            name=name, display_name=f"Banking77 configuration {index + 1}",
            experiment_name="banking77-guided-exercise", code=str(bundle),
            is_deterministic=False,
            command=f'python -u -m banking77.azure_worker --index {index} '
                    '--output "${{outputs.results}}"',
            environment=Environment(
                image=self.environment["image"],
                # Per bundle: a recovery may explicitly switch the torch build.
                conda_file=json.loads((bundle / "environment.json").read_text(encoding="utf-8"))),
            resources=JobResourceConfiguration(instance_type=compute["instance_type"],
                                               instance_count=1),
            queue_settings=QueueSettings(job_tier="Standard"),
            identity=UserIdentityConfiguration(),
            outputs={"results": Output(type="uri_folder", mode="rw_mount")},
            environment_variables={**manifest["environment"]["thread_settings"],
                                   "PYTHONUNBUFFERED": "1"},
        )
        # Omitting compute explicitly uses serverless (not a named shared cluster).
        try:
            returned = self.client.jobs.create_or_update(job)
        except AzureError as error:
            raise RuntimeError(f"Azure submission failed for {name}: {error}") from error
        return self._info(returned)

    @staticmethod
    def _info(job) -> dict:
        output = (job.outputs or {}).get("results")
        return {"name": job.name, "status": job.status,
                "studio_url": job.studio_url,
                "output_uri": getattr(output, "path", None),
                "error": str(job.error) if getattr(job, "error", None) else None}

    def inspect(self, name: str) -> dict:
        from azure.core.exceptions import AzureError, ResourceNotFoundError
        try:
            return self._info(self.client.jobs.get(name))
        except ResourceNotFoundError as error:
            raise AzureJobNotFoundError(f"Azure job {name} was not found: {error}") from error
        except AzureError as error:
            raise RuntimeError(f"Azure status failed for {name}: {error}") from error

    def download(self, name: str, destination: Path) -> Path:
        from azure.core.exceptions import AzureError
        try:
            self.client.jobs.download(name, download_path=str(destination), output_name="results")
        except AzureError as error:
            raise RuntimeError(f"Azure output download failed for {name}: {error}") from error
        return destination / "named-outputs" / "results"

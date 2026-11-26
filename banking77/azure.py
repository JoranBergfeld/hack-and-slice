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
    RESULT_FILES, capture, validate_manifest_results, validate_results, write_chart,
)
from banking77.artifacts import file_sha256, write_json
from banking77.azure_settings import job_name, torch_build, torch_device

RECOVERABLE = ("Failed", "Canceled", "SubmissionFailed")


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
    manifest = capture(arguments.configurations, arguments.prepared,
                       arguments.output, examples=arguments.examples)
    output = arguments.output.resolve()
    manifest.update(
        target="azure", device=torch_device(arguments.azure_device),
        status="submitting", results=[None] * len(manifest["configurations"]), jobs=[],
        azure={"subscription": arguments.subscription, "resource_group": arguments.resource_group,
               "workspace": arguments.workspace, "device": arguments.azure_device,
               "instance_type": arguments.instance_type, "tier": "Standard"},
    )
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
                        raise FileExistsError(f"{run} already exists but is not a validated result")
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


def recover_azure(arguments, *, service: AzureService | None = None) -> None:
    """Recover one failed job or completed job whose collected output was rejected."""
    from banking77.compare import expected_attempt
    from banking77.runner import LATEST_SNAPSHOT, load_snapshot
    output = arguments.launch.resolve()
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
    if job["status"] != "SubmissionFailed":
        job = service.inspect(arguments.job)
        if job["name"] != arguments.job:
            raise ValueError("Azure status returned a different job identifier")
        manifest["jobs"][index] = job
    if (job["status"] not in RECOVERABLE
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
    if bundle.exists():
        raise FileExistsError(f"{bundle} already exists")
    with tempfile.TemporaryDirectory(prefix="recover-", dir=output) as temporary:
        snapshot, entry = None, {
            "configuration_index": index, "mode": "resume" if arguments.resume else "restart",
            "attempt": attempt, "failed_job": arguments.job, "job_name": name,
            "device": device, "compute": compute, "training_environment": environment,
            "from_device": previous["device"], "resumed_after_epoch": None,
            "snapshot_sha256": None,
        }
        if arguments.resume:
            if job["status"] == "SubmissionFailed":
                raise ValueError(f"{arguments.job} never started; use --restart")
            try:
                downloaded = service.download(arguments.job, Path(temporary) / "download")
                snapshot = downloaded / LATEST_SNAPSHOT
                state = load_snapshot(snapshot)
            except (OSError, RuntimeError, ValueError, KeyError, EOFError,
                    pickle.UnpicklingError) as error:
                raise ValueError(f"{arguments.job} has no usable latest resumable snapshot "
                                 f"({error}); use --restart to train it again from epoch 1"
                                 ) from error
            if (state["identity"] != {"launch_id": manifest["launch_id"],
                                      "configuration_index": index}
                    or state["configuration"] != manifest["configurations"][index]
                    or state["prepared_sha256"] != manifest["prepared"]["sha256"]):
                raise ValueError(f"{arguments.job}: latest snapshot belongs to a different "
                                 "launch, configuration or prepared data")
            best = state["best"]
            entry.update(resumed_after_epoch=state["completed_epoch"],
                         from_device=state["device"], snapshot_sha256=file_sha256(snapshot))
            print(f"Latest resumable snapshot: after epoch {state['completed_epoch']} on "
                  f"{state['device']}; best validation {best['validation_accuracy']:.4f} "
                  f"at epoch {best['epoch']}.", flush=True)
            if state["device"] != device:
                print(f"Continuing on {device} after {state['device']}: valid, but not "
                      "guaranteed bit-for-bit identical to an uninterrupted run.", flush=True)
        shutil.copytree(output / "submission", bundle)
        if snapshot is not None:
            shutil.copy2(snapshot, bundle / LATEST_SNAPSHOT)
    write_json(bundle / "recovery.json", entry)
    write_json(bundle / "environment.json", environment["conda"])
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
        from azure.core.exceptions import AzureError
        try:
            return self._info(self.client.jobs.get(name))
        except AzureError as error:
            raise RuntimeError(f"Azure status failed for {name}: {error}") from error

    def download(self, name: str, destination: Path) -> Path:
        from azure.core.exceptions import AzureError
        try:
            self.client.jobs.download(name, download_path=str(destination), output_name="results")
        except AzureError as error:
            raise RuntimeError(f"Azure output download failed for {name}: {error}") from error
        return destination / "named-outputs" / "results"

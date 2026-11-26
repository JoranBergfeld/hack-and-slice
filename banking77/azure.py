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
    RESULT_FILES, _chart, _hash, _validate_results, _write, capture, validate_results,
)


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


def launch_azure(arguments, *, service: AzureService | None = None) -> None:
    for option in ("azure_device", "instance_type", "subscription", "resource_group", "workspace"):
        if not getattr(arguments, option):
            raise ValueError(f"--{option.replace('_', '-')} is required for Azure; no fallback")
    manifest = capture(arguments.configurations, arguments.prepared,
                       arguments.output, examples=arguments.examples)
    output = arguments.output.resolve()
    manifest.update(
        target="azure", device="cuda" if arguments.azure_device == "gpu" else "cpu",
        status="submitting", results=[None] * len(manifest["configurations"]), jobs=[],
        azure={"subscription": arguments.subscription, "resource_group": arguments.resource_group,
               "workspace": arguments.workspace, "device": arguments.azure_device,
               "instance_type": arguments.instance_type, "tier": "Standard"},
    )
    # Capture the exact environment definition alongside the submitted code.
    software = manifest["software"]
    torch_version = software["torch"].split("+")[0]
    build = "cu124" if arguments.azure_device == "gpu" else "cpu"
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
    manifest["training_environment"] = {
        "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        "conda": {"name": "banking77", "channels": ["conda-forge"],
                  "dependencies": [
                      f"python={software['python']}", "pip",
                      {"pip": [f"--extra-index-url https://download.pytorch.org/whl/{build}",
                               f"torch=={torch_version}+{build}",
                               f"scikit-learn=={software['scikit-learn']}", *extra_packages]},
                  ]},
    }
    _write(output / "launch.json", manifest)
    bundle = output / "submission"
    shutil.copytree(output / "source", bundle)
    shutil.copytree(output / "prepared", bundle / "prepared")
    _write(bundle / "launch.json", manifest)
    _write(bundle / "environment.json", manifest["training_environment"]["conda"])
    print(f"Target: Azure ML {arguments.azure_device.upper()}, "
          f"{arguments.instance_type}, Standard tier", flush=True)
    try:
        service = service or MLService(manifest["azure"], manifest["training_environment"])
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        manifest.update(status="setup_failed", error=str(error))
        _write(output / "launch.json", manifest)
        raise RuntimeError(f"Azure setup failed: {error}; install requirements-azure.txt "
                           "and authenticate with az login") from error
    failures = []
    for index in range(len(manifest["configurations"])):
        name = f"banking77-{manifest['launch_id']}-{index}"
        job = {"name": name, "status": "Submitting", "error": None}
        manifest["jobs"].append(job)
        _write(output / "launch.json", manifest)
        try:
            job = service.submit(bundle, name, index, manifest["azure"])
        except (RuntimeError, OSError, ValueError) as error:
            job.update(status="SubmissionFailed", error=str(error))
            failures.append(str(error))
        manifest["jobs"][index] = job
        _write(output / "launch.json", manifest)
        _show(job)
    manifest["status"] = "submission_failed" if failures else "submitted"
    _write(output / "launch.json", manifest)
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
        except (RuntimeError, OSError, ValueError) as error:
            job = {**previous, "error": str(error)}
            failures.append(str(error))
        manifest["jobs"][index] = job
        _show(job)
        if job["status"] in ("Failed", "Canceled", "SubmissionFailed"):
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
                        "sha256": {name: _hash(downloaded / name) for name in RESULT_FILES},
                        "software": record["software"],
                    }
                    _validate_results(output, candidate, allow_partial=True,
                                      pending_run=(index, downloaded))
                    run = output / "runs" / str(index)
                    run.parent.mkdir(exist_ok=True)
                    if run.exists():
                        raise FileExistsError(f"{run} already exists but is not a validated result")
                    downloaded.rename(run)
                    manifest["results"][index] = candidate["results"][index]
                    progress = run / "progress.log"
                    if progress.is_file():
                        print(progress.read_text(encoding="utf-8"), flush=True)
            except (ValueError, OSError, RuntimeError, KeyError, TypeError, IndexError,
                    EOFError, pickle.UnpicklingError) as error:
                job["collection_error"] = str(error)
                failures.append(f"{job['name']}: {error}")
                print(f"  Collection error: {error}", flush=True)
        _write(output / "launch.json", manifest)
    count = sum(result is not None for result in manifest["results"])
    if collect:
        if count == len(manifest["configurations"]):
            manifest["status"] = "completed"
            _write(output / "launch.json", manifest)
            _chart(output, validate_results(output))
        print(f"Collected {count}/{len(manifest['configurations'])} configurations. "
              "Selection requires every valid matching result.", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


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
            environment=Environment(image=self.environment["image"],
                                    conda_file=self.environment["conda"]),
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

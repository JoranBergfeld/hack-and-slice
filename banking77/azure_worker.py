"""Run one configuration inside the uploaded capture; not a participant solution."""
import argparse
import json
from pathlib import Path
import platform
import shutil

import torch

from banking77.artifacts import file_sha256, write_json
from banking77.azure_settings import job_name as azure_job_name
from banking77.cli import cli_errors
from banking77.runner import (
    LATEST_SNAPSHOT, PREPARED_FILES, TrainingConfiguration, run_training,
)

RECOVERY = "recovery.json"


@cli_errors("Azure training", "03-train-locally.md")
def main() -> int:
    parser = argparse.ArgumentParser(description="Azure job entry point for captured code.")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads(Path("launch.json").read_text(encoding="utf-8"))
    index = arguments.index
    if not 0 <= index < len(manifest["configurations"]):
        raise ValueError("configuration index is outside this launch")
    for name, digest in manifest["source"]["sha256"].items():
        path = Path(name).resolve()
        if not path.is_relative_to(Path.cwd().resolve()) or file_sha256(path) != digest:
            raise ValueError(f"mismatched captured code: {name}")
    for name in PREPARED_FILES:
        if file_sha256(Path("prepared") / name) != manifest["prepared"]["sha256"][name]:
            raise ValueError(f"mismatched captured data: {name}")
    device = torch.device(manifest["device"])
    compute, environment = manifest["azure"], manifest["training_environment"]
    job_name = azure_job_name(manifest["launch_id"], index)
    recovery, snapshot = None, None
    if Path(RECOVERY).is_file():
        # An explicit participant recovery of this configuration only.
        recovery = json.loads(Path(RECOVERY).read_text(encoding="utf-8"))
        if recovery["configuration_index"] != index:
            raise ValueError("recovery belongs to a different configuration")
        device = torch.device(recovery["device"])
        compute, environment = recovery["compute"], recovery["training_environment"]
        job_name = recovery["job_name"]
        if recovery["mode"] == "resume":
            snapshot = Path(LATEST_SNAPSHOT)
            if file_sha256(snapshot) != recovery["snapshot_sha256"]:
                raise ValueError("mismatched recovery snapshot")
            # Keep the latest snapshot with this attempt even if it fails early.
            arguments.output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot, arguments.output / LATEST_SNAPSHOT)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Azure GPU was selected but CUDA is unavailable; no CPU fallback")
    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / "progress.log").open("w", encoding="utf-8") as progress:
        def log(message):
            print(message, flush=True)
            progress.write(message + "\n")
            progress.flush()

        record = run_training(
            TrainingConfiguration(**manifest["configurations"][index]),
            Path("prepared"), arguments.output, device=device,
            examples=manifest["examples"], log=log,
            target=f"Azure ML {compute['device'].upper()}",
            resume=snapshot,
            identity={"launch_id": manifest["launch_id"], "configuration_index": index},
        )
    record.update(
        provenance={"launch_id": manifest["launch_id"], "configuration_index": index,
                    "source": manifest["source"], "environment": manifest["environment"]},
        training_environment=environment, compute=compute, job_name=job_name,
        recovery=recovery,
        runtime_environment={"platform": platform.platform(),
                             "architecture": platform.machine(),
                             "torch_threads": torch.get_num_threads(),
                             "cuda": torch.version.cuda},
    )
    write_json(arguments.output / "run.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

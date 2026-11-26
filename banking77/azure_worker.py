"""Run one configuration inside the uploaded capture; not a participant solution."""
import argparse
import json
from pathlib import Path
import platform
import sys

import torch

from banking77.compare import _hash, _write
from banking77.runner import PREPARED_FILES, TrainingConfiguration, run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Azure job entry point for captured code.")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        manifest = json.loads(Path("launch.json").read_text(encoding="utf-8"))
        index = arguments.index
        if not 0 <= index < len(manifest["configurations"]):
            raise ValueError("configuration index is outside this launch")
        for name, digest in manifest["source"]["sha256"].items():
            path = Path(name).resolve()
            if not path.is_relative_to(Path.cwd().resolve()) or _hash(path) != digest:
                raise ValueError(f"mismatched captured code: {name}")
        for name in PREPARED_FILES:
            if _hash(Path("prepared") / name) != manifest["prepared"]["sha256"][name]:
                raise ValueError(f"mismatched captured data: {name}")
        device = torch.device(manifest["device"])
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
                target=f"Azure ML {manifest['azure']['device'].upper()}",
            )
        record.update(
            provenance={"launch_id": manifest["launch_id"], "configuration_index": index,
                        "source": manifest["source"], "environment": manifest["environment"]},
            training_environment=manifest["training_environment"], compute=manifest["azure"],
            job_name=f"banking77-{manifest['launch_id']}-{index}",
            runtime_environment={"platform": platform.platform(),
                                 "architecture": platform.machine(),
                                 "torch_threads": torch.get_num_threads(),
                                 "cuda": torch.version.cuda},
        )
        _write(arguments.output / "run.json", record)
    except (NotImplementedError, ValueError, OSError, RuntimeError, KeyError,
            AttributeError, TypeError, IndexError, ZeroDivisionError) as error:
        print(f"Azure training failed: {error}. "
              "See docs/challenges/03-train-locally.md.", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

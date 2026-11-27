"""Offline external Azure boundary; submission copies code but never trains."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


class FakeAzure:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(exist_ok=True)

    def submit(self, bundle, name, index, compute):
        control = self.root / "control.json"
        if control.exists() and index in json.loads(control.read_text()).get("fail_submit", []):
            raise RuntimeError(f"synthetic submission quota error for configuration {index}")
        job = self.root / name
        shutil.copytree(bundle, job / "code")
        info = {"name": name, "status": "Queued",
                "studio_url": f"https://azure.invalid/jobs/{name}",
                "output_uri": f"azureml://fake/{name}/results",
                "error": None, "compute": compute, "index": index}
        (job / "info.json").write_text(json.dumps(info))
        return info

    def inspect(self, name):
        return json.loads((self.root / name / "info.json").read_text())

    def download(self, name, destination):
        shutil.copytree(self.root / name / "results", destination)
        return destination

    def interrupt(self, name, *, after_epochs, emulate_gpu_on_cpu=False):
        """Hard-kill the job's worker during epoch after_epochs + 1, like a preemption.

        emulate_gpu_on_cpu runs a GPU job's captured worker on this CPU-only host;
        its durable outputs stand in for those of a real GPU node.
        """
        job = self.root / name
        info = self.inspect(name)
        code = job / "code"
        if emulate_gpu_on_cpu:
            code = job / "emulated-gpu-node"
            shutil.copytree(job / "code", code)
            manifest = json.loads((code / "launch.json").read_text())
            manifest["device"] = "cpu"
            (code / "launch.json").write_text(json.dumps(manifest))
        script = (
            "import os, sys\n"
            "import banking77.training as training\n"
            "original, calls = training.train_epoch, [0]\n"
            "def preempted(*arguments):\n"
            "    calls[0] += 1\n"
            f"    if calls[0] > {after_epochs}:\n"
            "        os._exit(137)\n"
            "    return original(*arguments)\n"
            "training.train_epoch = preempted\n"
            "from banking77.azure_worker import main\n"
            f"sys.argv = ['worker', '--index', '{info['index']}', '--output', "
            f"{str(job / 'results')!r}]\n"
            "raise SystemExit(main())\n"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=code,
                                capture_output=True, text=True)
        if result.returncode != 137:
            raise RuntimeError(f"expected an interrupted worker: {result.stderr}")
        snapshot = job / "results" / "latest-snapshot.pt"
        if emulate_gpu_on_cpu and snapshot.is_file():
            import torch
            state = torch.load(snapshot, weights_only=True)
            state["device"] = "cuda"
            for entry in state["history"]:
                entry["device"] = "cuda"
            state["random_state"]["torch_cuda"] = [torch.zeros(1, dtype=torch.uint8)]
            torch.save(state, snapshot)
        info.update(status="Failed",
                    error=f"synthetic node preemption during epoch {after_epochs + 1}")
        (job / "info.json").write_text(json.dumps(info))

    def fail(self, name, error):
        info = self.inspect(name)
        info.update(status="Failed", error=error)
        (self.root / name / "info.json").write_text(json.dumps(info))

    def complete(self, name):
        job = self.root / name
        info = self.inspect(name)
        result = subprocess.run(
            [sys.executable, "-m", "banking77.azure_worker", "--index",
             str(info["index"]), "--output", str(job / "results")],
            cwd=job / "code", capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        info["status"] = "Completed"
        (job / "info.json").write_text(json.dumps(info))

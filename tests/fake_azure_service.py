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

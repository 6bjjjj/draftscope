"""Install the built wheel in an isolated environment without ML dependencies."""

import json
from pathlib import Path
import subprocess
import tempfile
import venv


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("draftscope-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one built DraftScope wheel")
    with tempfile.TemporaryDirectory(prefix="draftscope-wheel-") as folder:
        folder = Path(folder)
        venv.create(folder / "env", with_pip=True)
        candidates = [folder / "env/Scripts/python.exe", folder / "env/bin/python"]
        python = next(p for p in candidates if p.exists())
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(wheels[0])],
            check=True,
            cwd=folder,
        )
        script = "import draftscope,sys; assert 'torch' not in sys.modules; print(draftscope.__version__)"
        subprocess.run([str(python), "-I", "-c", script], check=True, cwd=folder)
        result = subprocess.run(
            [str(python), "-I", "-m", "draftscope", "plan"],
            check=True,
            cwd=folder,
            capture_output=True,
            text=True,
        )
        value = json.loads(result.stdout)
        if value["gpu_tested"] or value["quality_accepted"] is not None:
            raise RuntimeError("Planning invented validation evidence")
        print("Clean wheel import and no-Torch planning passed")


if __name__ == "__main__":
    main()

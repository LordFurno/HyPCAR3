"""Compatibility entrypoint for the original HyPCAR project layout.

The cleaned public layout keeps the implementation in
``pipeline/physics_finetune.py``. Running this historical path forwards to the
new script.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "pipeline" / "physics_finetune.py"
    runpy.run_path(str(target), run_name="__main__")

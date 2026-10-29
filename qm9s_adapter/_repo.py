"""
The simulator imports itself as `ters_img_simulator`, but this checkout may be named
differently (e.g. ters-simulator). Registering the repository root under that name makes
the imports work from any checkout, without installing anything.
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def bind_package():
    if "ters_img_simulator" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "ters_img_simulator", REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ters_img_simulator"] = module
    spec.loader.exec_module(module)

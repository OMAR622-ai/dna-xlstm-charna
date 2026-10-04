"""Make the repo's leaf modules importable without mamba_ssm / flash_attn / lightning.

Registers empty stand-ins for packages whose __init__.py pulls in those dependencies, so
`models.charna`, `models.llama`, `models.caduceus.tokenization_caduceus` and
`src.dataloaders.*` import on a CPU-only machine. Nothing in the repo is modified.
(Same approach as sample_run/bootstrap.py.)
"""

import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _stand_in(name, path):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules[name] = mod


def install():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _stand_in("models", REPO_ROOT / "models")
    _stand_in("models.caduceus", REPO_ROOT / "models" / "caduceus")
    _stand_in("src", REPO_ROOT / "src")
    _stand_in("src.dataloaders", REPO_ROOT / "src" / "dataloaders")


install()

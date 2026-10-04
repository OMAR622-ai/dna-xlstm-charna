"""Run the repo's train.py on machines without mamba_ssm / flash_attn / causal_conv1d.

train.py imports these packages at module level (via the Caduceus model, the Mamba downstream
wrapper and flash_attn's GenerationMixin) even when they are not used. They need CUDA builds
that pre-Ampere GPUs such as the GTX 1080 Ti cannot run. For each of them that is *not*
installed, this launcher registers a placeholder package whose attributes are inert classes,
so imports succeed; actually using a Mamba/Caduceus model would fail loudly. Installed packages
are left untouched, and no repo file is modified.

Usage (same Hydra overrides as train.py, run from the repo root):
  python scripts_charna/train_charna.py experiment=hg38/hg38 model=charna ...
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OPTIONAL_CUDA_PACKAGES = ("mamba_ssm", "flash_attn", "causal_conv1d", "selective_scan_cuda")


class _Unavailable:
    """Placeholder for a symbol from a package that is not installed."""

    _missing = "an optional CUDA package"

    def __init__(self, *args, **kwargs):
        raise ImportError(f"{type(self).__name__} comes from {self._missing}, which is not installed")


class _PlaceholderModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (_Unavailable,), {"_missing": self.__name__.split(".")[0]})
        setattr(self, name, placeholder)
        return placeholder


class _PlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, roots):
        self.roots = set(roots)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _PlaceholderModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


def install_placeholders():
    missing = [name for name in OPTIONAL_CUDA_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        print(f"[train_charna] placeholders for missing optional packages: {', '.join(missing)}")
        sys.meta_path.append(_PlaceholderFinder(missing))


if __name__ == "__main__":
    install_placeholders()
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    sys.argv[0] = str(REPO_ROOT / "train.py")
    import train  # noqa: E402

    train.main()

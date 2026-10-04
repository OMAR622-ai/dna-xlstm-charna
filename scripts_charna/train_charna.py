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


# Names that repo modules import unconditionally at module level. Every other lookup raises
# AttributeError (-> ImportError for `from x import y`), so the repo's own try/except fallbacks
# (e.g. torch CrossEntropyLoss instead of flash_attn's) still take effect.
PLACEHOLDER_NAMES = {
    "flash_attn.utils.generation": {"GenerationMixin"},
    "flash_attn.modules.block": {"Block"},
    "flash_attn.modules.embedding": {"GPT2Embeddings", "ParallelGPT2Embeddings"},
    "flash_attn.modules.mha": {"MHA", "ParallelMHA"},
    "flash_attn.modules.mlp": {"Mlp", "FusedMLP", "ParallelFusedMLP"},
    "flash_attn.utils.distributed": {"sync_shared_params", "all_gather_raw"},
    "mamba_ssm.models.config_mamba": {"MambaConfig"},
    "mamba_ssm.models.mixer_seq_simple": {"MixerModel", "_init_weights"},
    "mamba_ssm.modules.mamba_simple": {"Mamba", "Block"},
}


class _PlaceholderModule(types.ModuleType):
    def __getattr__(self, name):
        if name not in PLACEHOLDER_NAMES.get(self.__name__, ()):
            raise AttributeError(f"module {self.__name__!r} is a placeholder without {name!r}")
        placeholder = type(name, (_Unavailable,), {"_missing": self.__name__.split(".")[0]})
        setattr(self, name, placeholder)
        return placeholder


class _PlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, roots):
        # Only the modules on the path to an allow-listed name exist; anything else is a
        # normal ModuleNotFoundError, so `try: from flash_attn.x import y` falls back as intended.
        self.modules = {
            ".".join(mod.split(".")[:i])
            for mod in PLACEHOLDER_NAMES if mod.split(".")[0] in roots
            for i in range(1, mod.count(".") + 2)
        } | set(roots)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _PlaceholderModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


def ensure_cuda_home():
    """The xLSTM sLSTM package calls cpp_extension.include_paths(cuda=True) at import time, which
    fails without CUDA_HOME. If no CUDA toolkit is configured, point it at the CUDA runtime that
    pip installs with torch; it is only read to set CUDA_LIB for an sLSTM build we never run."""
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    spec = importlib.util.find_spec("nvidia.cuda_runtime")
    if spec is not None and spec.submodule_search_locations:
        os.environ["CUDA_HOME"] = list(spec.submodule_search_locations)[0]
        print(f"[train_charna] CUDA_HOME not set; using {os.environ['CUDA_HOME']} (no compilation happens)")


def install_placeholders():
    missing = [name for name in OPTIONAL_CUDA_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        print(f"[train_charna] placeholders for missing optional packages: {', '.join(missing)}")
        sys.meta_path.append(_PlaceholderFinder(missing))


if __name__ == "__main__":
    install_placeholders()
    ensure_cuda_home()
    sys.argv[0] = os.path.abspath(sys.argv[0])  # Hydra changes cwd before DDP spawns ranks > 0
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    # sys.argv[0] stays this launcher: Lightning's DDP re-runs sys.argv[0] for ranks > 0, and those
    # processes need the placeholders too (runpy.run_path would rewrite sys.argv[0], so exec instead).
    # Executing as __main__ with train.py's filename lets Hydra resolve config_path="configs".
    train_py = REPO_ROOT / "train.py"
    code = compile(train_py.read_text(), str(train_py), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(train_py), "__builtins__": __builtins__})

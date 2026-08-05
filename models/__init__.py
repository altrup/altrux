import sys
import types

# mamba_ssm/__init__.py unconditionally imports legacy Mamba-1 ops
# (selective_scan_fn, mamba_inner_fn), which need the compiled
# `selective_scan_cuda` extension -- nothing in this repo calls those
# functions (every model here only uses Mamba2, via MambaLMHeadModel/
# MambaConfig, plus its own manual _mixer_step). On this machine that
# extension fails to import (a ROCm runtime library naming/ABI mismatch
# between whatever torch+ROCm build mamba-ssm's wheel was compiled against
# and the current one -- unrelated to anything in this repo and not worth
# chasing further since the functions are unused). Stubbing it out in
# sys.modules before mamba_ssm is ever imported makes Python's import system
# skip the real (broken) import entirely. This must run before any
# `import mamba_ssm` anywhere, which is why it lives here: every model
# import goes through this package's __init__ first.
if "selective_scan_cuda" not in sys.modules:
    sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")

# Same trick for Mamba-3 (mamba-ssm >= 2.3): mamba_ssm/__init__.py imports
# Mamba3, whose module imports tilelang -> tvm, which dlopens
# libtorch_cuda.so at import time -- an OSError on a ROCm torch build, which
# the upstream `except ImportError` guard does not catch. Nothing in this
# repo uses Mamba3. Stub only where the real import would break (a CUDA box
# gets the real class).
import torch

if torch.version.hip is not None and "mamba_ssm.modules.mamba3" not in sys.modules:
    _mamba3 = types.ModuleType("mamba_ssm.modules.mamba3")
    _mamba3.Mamba3 = None  # unused in this repo; import-time placeholder only
    sys.modules["mamba_ssm.modules.mamba3"] = _mamba3

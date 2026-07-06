"""Override for pyinstaller-hooks-contrib's hook-onnxruntime.py.

The stock hook does `collect_dynamic_libs("onnxruntime")`, which grabs every
DLL in onnxruntime/capi/ — including onnxruntime_providers_cuda.dll, a
~244MB file. scanner.py only ever requests CPUExecutionProvider, so that
DLL is pure dead weight in the shipped exe. Drop it here instead.
"""

import os
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = [
    (src, dest) for src, dest in collect_dynamic_libs("onnxruntime")
    if "cuda" not in os.path.basename(src).lower()
    and "tensorrt" not in os.path.basename(src).lower()
]

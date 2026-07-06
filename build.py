"""build.py — builds BuckshotOverlay.exe from overlay.py.

Usage:
    python build.py

Bundles only what the app actually needs at runtime: overlay.py + its
imports (scanner.py, ai_engine.py), best.onnx, and pics/. Does NOT bundle
best.pt, train.py, download_dataset.py, main.py, or anything under
scenario/ / tests/ — none of that is imported by overlay.py, so
PyInstaller's dependency scan never touches them.

The --exclude-module flags below matter: onnxruntime ships optional
quantization/conversion submodules (onnxruntime.quantization, .tools,
.transformers) that reference torch/scipy/pandas/matplotlib/etc. purely
for those unused features. PyInstaller's static analysis can't tell those
imports are dead code, so without the excludes it drags the entire
training-side ML stack into the exe (confirmed: turned a ~50MB build into
a ~2.9GB one). None of this is reachable at actual runtime — verified by
diffing sys.modules before/after `import onnxruntime`.
"""

import PyInstaller.__main__

EXCLUDES = [
    # onnxruntime's own unused conversion/quantization tooling
    "onnxruntime.quantization",
    "onnxruntime.tools",
    "onnxruntime.transformers",
    "onnxruntime.training",
    # heavy training-side stack pulled in only via the above, never used
    "torch", "torchvision", "ultralytics", "ultralytics.thop",
    "scipy", "pandas", "matplotlib", "sympy", "mpmath", "networkx", "triton",
    "onnx", "onnxslim", "ml_dtypes",
    "safetensors", "einops", "cutlass",
]

args = [
    "overlay.py",
    "--name=BuckshotOverlay",
    "--onefile",
    "--windowed",
    "--add-data=best.onnx;.",
    "--add-data=pics;pics",
    "--additional-hooks-dir=pyinstaller_hooks",
    "--noconfirm",
]
for mod in EXCLUDES:
    args.append(f"--exclude-module={mod}")

PyInstaller.__main__.run(args)

#!/usr/bin/env python3
"""
Convert LaMa ONNX model from FP32 to FP16.

Usage:
    pip install onnx onnxconverter-common
    python scripts/convert_lama_fp16.py

Input:  app/src/main/assets/lama.onnx      (FP32, ~198 MB)
Output: app/src/main/assets/lama_fp16.onnx  (FP16, ~99 MB)

After conversion, rebuild the app. LamaInpainter will auto-detect and
prefer the FP16 model, falling back to FP32 if not found.
"""

import os
import sys

try:
    import onnx
    from onnxconverter_common import float16
except ImportError:
    print("Install dependencies first:")
    print("  pip install onnx onnxconverter-common")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PROJECT_ROOT, "app", "src", "main", "assets")

INPUT_PATH = os.path.join(ASSETS_DIR, "lama.onnx")
OUTPUT_PATH = os.path.join(ASSETS_DIR, "lama_fp16.onnx")

if not os.path.exists(INPUT_PATH):
    print(f"Error: {INPUT_PATH} not found")
    sys.exit(1)

print(f"Loading {INPUT_PATH} ...")
model = onnx.load(INPUT_PATH)

print("Converting to FP16 ...")
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,
)

print(f"Saving {OUTPUT_PATH} ...")
onnx.save(model_fp16, OUTPUT_PATH)

input_size = os.path.getsize(INPUT_PATH) / (1024 * 1024)
output_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"Done. {input_size:.1f} MB -> {output_size:.1f} MB ({output_size/input_size*100:.0f}%)")
# TFLite Export on Windows

Native TFLite export via Ultralytics (`format='litert'`) is only supported on
Linux x86 and macOS. On Windows, `export_all.py` automatically falls back to a
two-step pipeline that produces the same files:

```
best.pt  ──►  ONNX (static, opset=12, batch=1)  ──►  TFLite (FP16 / FP32)
              via Ultralytics                             via onnx2tf
```

The whole thing is driven by `export_all.py --no-copy` so you can run:

```bash
# from the workspace root
cd nail_ai_workspace
venv311\Scripts\python.exe export_all.py --run Nail_v1i_yolov11_20260723_070629
```

and get both `nail_seg.onnx` and `nail_seg_float16.tflite` under
`mobile_model/<run_name>/`.

## TensorFlow / TFLite layout differences

The ONNX and TFLite models share the same YOLO weights but have slightly
different tensor layouts. Mobile inference code needs to be aware of this:

| Tensor        | ONNX shape     | TFLite shape    | Notes                |
|---------------|----------------|-----------------|----------------------|
| Input         | `[1, 3, 416, 416]` | `[1, 416, 416, 3]` | NCHW vs NHWC |
| Output 0      | `[1, 37, 3549]` | `[1, 37, 3549]` | same |
| Output 1 (proto) | `[1, 32, 104, 104]` | `[1, 104, 104, 32]` | channels last |

So a TFLite client must:
1. Feed the input as NHWC (transpose from NCHW).
2. Transpose the mask-prototype output back to NCHW before reconstructing
   the per-pixel mask.

## Required packages

Pinned in `requirements.txt`:

- `tensorflow` (>= 2.16) — the actual TFLite compiler
- `onnx2tf` (>= 1.28) — fallback ONNX → TFLite converter
- `tf_keras`, `onnx-graphsurgeon`, `onnx-simplifier`, `sng4onnx`, `ai-edge-litert`
  — transitive deps that Windows numpy 2.x users will need

Note: `tensorflow` only ships wheels for Python 3.9 - 3.12. On Python 3.13+ the
TFLite pipeline cannot run; the desktop app + ONNX Runtime path still works.

## Cleaning up

The Windows pipeline writes a `_onnx2tf_work/` subdirectory plus a
`calibration_image_sample_data_*.npy` cache during conversion. Both are
auto-cleaned on success. If conversion fails mid-way, you can delete them by
hand:

```bash
rm -rf mobile_model/<run_name>/_onnx2tf_work
```

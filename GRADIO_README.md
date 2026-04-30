# LingBot-Map — Gradio UI

A simple browser UI for the LingBot-Map streaming 3D reconstructor. Drop in a
video or a folder of images, hit run, and view the reconstructed point cloud
inline (and download the GLB).

## What you get

- Three input modes: **video**, **uploaded images**, or one of the built-in
  `example/` folders
- Auto-download of the checkpoint from HuggingFace
  (`robbyant/lingbot-map`)
- Sensible defaults for first run (120 frames cap, sky mask off, SDPA backend
  so FlashInfer is not required)
- Result rendered in-browser via Gradio's `Model3D` component, and downloadable
  as a `.glb`

## First-run setup (Windows, RTX-class GPU)

```bat
setup.bat
run_app.bat
```

That creates a `.venv` (Python 3.10), installs PyTorch 2.8.0 + CUDA 12.8, the
`lingbot-map` package + `[vis]` extras, Gradio, and `onnxruntime`.

To launch with a public share link:

```bat
run_app.bat --share
```

## First-run setup (manual, any platform)

```bash
uv venv --python 3.10 .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -e ".[vis]" gradio onnxruntime
.venv/Scripts/python.exe app.py
```

The UI starts at <http://localhost:7860>.

## Tips

- **First run is slow** — the model is loaded into memory and cached for
  subsequent runs (changing the checkpoint, `use_sdpa`, or camera iters
  invalidates the cache).
- **Long videos**: cap with the *Max frames* field, or leave
  `keyframe_interval = 0` to let it auto-pick. >320 frames will trim KV cache
  by sub-sampling keyframes.
- **Outdoor scenes**: tick *Mask sky*. The first time, it downloads a small
  ONNX model from HuggingFace.
- **No FlashInfer? No problem.** SDPA is on by default. Untick only if you
  installed FlashInfer separately.
- **Checkpoints** live under `checkpoints/`. You can also paste a full path
  into the *Custom checkpoint path* box to use one you already have.

## Files added by this UI

- `app.py` — the Gradio app
- `setup.bat` / `run_app.bat` / `run_app.sh` — launchers
- `checkpoints/` — auto-created, holds downloaded `.pt` files

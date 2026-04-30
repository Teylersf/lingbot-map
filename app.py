"""Gradio UI for LingBot-Map streaming 3D reconstruction.

Usage:
    python app.py                                  # auto-downloads checkpoint
    python app.py --model_path /path/to/ckpt.pt    # uses local checkpoint
    python app.py --share                          # public Gradio link

The app accepts a video or a folder of images, runs streaming inference, and
renders the resulting point cloud as a downloadable GLB you can view in the
browser via Gradio's built-in Model3D viewer.
"""

import argparse
import glob
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import numpy as np
import torch

from demo import (
    load_images,
    load_model,
    postprocess,
    prepare_for_visualization,
)

REPO_ROOT = Path(__file__).resolve().parent
CKPT_DIR = REPO_ROOT / "checkpoints"
EXAMPLE_DIR = REPO_ROOT / "example"
CKPT_DIR.mkdir(exist_ok=True)

DEFAULT_HF_REPO = "robbyant/lingbot-map"
CHECKPOINT_OPTIONS = {
    "lingbot-map-long (recommended, long sequences)": "lingbot-map-long.pt",
    "lingbot-map (balanced)": "lingbot-map.pt",
    "lingbot-map-stage1": "lingbot-map-stage1.pt",
}

_MODEL_CACHE: dict = {}


class Args:
    """Argparse-compatible namespace built from Gradio inputs."""

    def __init__(self, **kwargs):
        self.image_size = 518
        self.patch_size = 14
        self.enable_3d_rope = True
        self.max_frame_num = 1024
        self.num_scale_frames = 8
        self.kv_cache_sliding_window = 64
        self.use_sdpa = True
        self.camera_num_iterations = 4
        self.mode = "streaming"
        self.model_path = None
        for k, v in kwargs.items():
            setattr(self, k, v)


def ensure_checkpoint(checkpoint_label: str, custom_path: str = "") -> str:
    """Resolve a checkpoint to a local path, downloading from HuggingFace if needed."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    filename = CHECKPOINT_OPTIONS.get(checkpoint_label)
    if filename is None:
        raise ValueError(f"Unknown checkpoint: {checkpoint_label}")

    local = CKPT_DIR / filename
    if local.is_file():
        return str(local)

    from huggingface_hub import hf_hub_download

    print(f"Downloading {filename} from {DEFAULT_HF_REPO}...")
    path = hf_hub_download(
        repo_id=DEFAULT_HF_REPO,
        filename=filename,
        local_dir=str(CKPT_DIR),
    )
    return path


def get_model(model_path: str, use_sdpa: bool, camera_num_iterations: int):
    key = (model_path, use_sdpa, camera_num_iterations)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Args(
        model_path=model_path,
        use_sdpa=use_sdpa,
        camera_num_iterations=camera_num_iterations,
        mode="streaming",
    )
    model = load_model(args, device)

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if model.aggregator is not None:
            model.aggregator = model.aggregator.to(dtype=dtype)

    _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = (model, device)
    return model, device


def _stage_inputs(video_file, image_files, image_folder_path):
    """Resolve UI inputs to either a video path or an image folder path."""
    if video_file:
        return {"video_path": video_file, "image_folder": None, "cleanup": None}

    if image_files:
        tmp = tempfile.mkdtemp(prefix="lingbot_uploads_")
        for i, f in enumerate(image_files):
            ext = os.path.splitext(f)[1] or ".jpg"
            shutil.copy(f, os.path.join(tmp, f"{i:06d}{ext}"))
        return {"video_path": None, "image_folder": tmp, "cleanup": tmp}

    if image_folder_path and os.path.isdir(image_folder_path):
        return {"video_path": None, "image_folder": image_folder_path, "cleanup": None}

    raise gr.Error("Provide a video, upload images, or pick an example folder.")


def reconstruct(
    video_file,
    image_files,
    image_folder_path,
    checkpoint_label,
    custom_ckpt_path,
    fps,
    first_k,
    keyframe_interval,
    mask_sky,
    conf_threshold,
    camera_num_iterations,
    use_sdpa,
    show_cameras,
    progress=gr.Progress(track_tqdm=True),
):
    if not torch.cuda.is_available():
        gr.Warning("No CUDA GPU detected — inference will be very slow on CPU.")

    progress(0.0, desc="Resolving inputs")
    staged = _stage_inputs(video_file, image_files, image_folder_path)

    progress(0.05, desc="Resolving checkpoint")
    ckpt_path = ensure_checkpoint(checkpoint_label, custom_ckpt_path)

    progress(0.1, desc="Loading model (cached after first run)")
    model, device = get_model(
        ckpt_path,
        use_sdpa=use_sdpa,
        camera_num_iterations=int(camera_num_iterations),
    )

    progress(0.2, desc="Loading frames")
    images, paths, resolved_folder = load_images(
        image_folder=staged["image_folder"],
        video_path=staged["video_path"],
        fps=int(fps),
        first_k=int(first_k) if first_k and int(first_k) > 0 else None,
    )
    if images.shape[0] < 2:
        raise gr.Error(f"Got only {images.shape[0]} frames — need at least 2.")

    images = images.to(device)
    num_frames = images.shape[0]
    info_lines = [f"Loaded {num_frames} frames at {tuple(images.shape[-2:])}."]

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    kf = int(keyframe_interval) if keyframe_interval and int(keyframe_interval) > 0 else None
    if kf is None:
        kf = max(1, (num_frames + 319) // 320) if num_frames > 320 else 1
    info_lines.append(f"keyframe_interval = {kf}")

    progress(0.3, desc=f"Running inference ({num_frames} frames)")
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=8,
            keyframe_interval=kf,
            output_device=torch.device("cpu"),
        )
    dt = time.time() - t0
    info_lines.append(f"Inference: {dt:.1f}s ({num_frames / max(dt, 1e-6):.1f} FPS)")

    progress(0.85, desc="Post-processing")
    images_for_post = predictions.get("images", images)
    predictions, images_cpu = postprocess(predictions, images_for_post)
    vis_pred = prepare_for_visualization(predictions, images_cpu)

    if "world_points_from_depth" not in vis_pred:
        from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
        vis_pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
            vis_pred["depth"], vis_pred["extrinsic"], vis_pred["intrinsic"]
        )

    progress(0.92, desc="Building GLB")
    from lingbot_map.vis import predictions_to_glb

    target_dir = None
    if mask_sky:
        target_dir = tempfile.mkdtemp(prefix="lingbot_skydir_")
        os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "sky_masks"), exist_ok=True)
        for p in paths:
            shutil.copy(p, os.path.join(target_dir, "images", os.path.basename(p)))

    scene = predictions_to_glb(
        vis_pred,
        conf_thres=float(conf_threshold),
        show_cam=bool(show_cameras),
        mask_sky=bool(mask_sky),
        target_dir=target_dir,
    )

    out_dir = tempfile.mkdtemp(prefix="lingbot_glb_")
    glb_path = os.path.join(out_dir, "reconstruction.glb")
    scene.export(glb_path)

    if staged["cleanup"]:
        shutil.rmtree(staged["cleanup"], ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    progress(1.0, desc="Done")

    info_lines.append(f"GLB saved: {glb_path}")
    return glb_path, glb_path, "\n".join(info_lines)


def list_example_folders():
    if not EXAMPLE_DIR.is_dir():
        return []
    return sorted(str(p) for p in EXAMPLE_DIR.iterdir() if p.is_dir())


def build_ui():
    with gr.Blocks(title="LingBot-Map") as demo:
        gr.Markdown(
            "## LingBot-Map — Streaming 3D Reconstruction\n"
            "Upload a video or images, run inference, and view the reconstructed "
            "point cloud (GLB) right in the browser."
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Video"):
                        video_in = gr.Video(label="Input video", sources=["upload"])
                    with gr.Tab("Images"):
                        images_in = gr.File(
                            label="Upload images",
                            file_count="multiple",
                            file_types=["image"],
                        )
                    with gr.Tab("Example folder"):
                        examples = list_example_folders()
                        folder_in = gr.Dropdown(
                            label="Built-in example",
                            choices=examples,
                            value=examples[0] if examples else None,
                        )

                with gr.Accordion("Model & inference", open=True):
                    ckpt_choice = gr.Dropdown(
                        label="Checkpoint (auto-downloaded from HuggingFace)",
                        choices=list(CHECKPOINT_OPTIONS.keys()),
                        value="lingbot-map-long (recommended, long sequences)",
                    )
                    custom_ckpt = gr.Textbox(
                        label="Custom checkpoint path (optional, overrides above)",
                        value="",
                        placeholder="/path/to/your.pt",
                    )
                    fps = gr.Slider(1, 30, value=10, step=1, label="Video sampling FPS")
                    first_k = gr.Number(
                        value=120,
                        precision=0,
                        label="Max frames (0 = all). Keep small for first run.",
                    )
                    keyframe_interval = gr.Number(
                        value=0,
                        precision=0,
                        label="keyframe_interval (0 = auto)",
                    )
                    camera_iters = gr.Slider(
                        1, 4, value=4, step=1, label="Camera refinement iterations"
                    )
                    use_sdpa = gr.Checkbox(
                        value=True,
                        label="Use SDPA (PyTorch native attention) — uncheck only if FlashInfer installed",
                    )

                with gr.Accordion("Visualization", open=True):
                    mask_sky = gr.Checkbox(value=False, label="Mask sky (outdoor scenes)")
                    conf_threshold = gr.Slider(
                        0, 95, value=50, step=1,
                        label="Confidence percentile filter (higher = stricter)",
                    )
                    show_cameras = gr.Checkbox(value=True, label="Show camera frustums")

                run_btn = gr.Button("Run reconstruction", variant="primary")

            with gr.Column(scale=1):
                model3d = gr.Model3D(
                    label="Reconstructed point cloud",
                    clear_color=[0.05, 0.05, 0.08, 1.0],
                    height=520,
                )
                glb_download = gr.File(label="Download GLB")
                info = gr.Textbox(label="Run info", lines=6)

        run_btn.click(
            reconstruct,
            inputs=[
                video_in, images_in, folder_in,
                ckpt_choice, custom_ckpt,
                fps, first_k, keyframe_interval,
                mask_sky, conf_threshold, camera_iters, use_sdpa, show_cameras,
            ],
            outputs=[model3d, glb_download, info],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()

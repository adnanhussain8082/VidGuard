
import argparse
import mimetypes
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from gradcam_utils import GradCAM, overlay_heatmap_on_rgb

# Shared preprocessing used for image and video frames.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_model(model_path="models/best_model-v3.pt"):
    """Load EfficientNet-B0 binary classifier on CPU."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def _predict_with_gradcam_from_rgb(rgb_image, model, gradcam=None):
    """
    Run prediction + Grad-CAM for one RGB image array.

    Returns:
        label (str), confidence (float), overlay_rgb (np.ndarray), heatmap (np.ndarray)
    """
    pil_image = Image.fromarray(rgb_image)
    input_tensor = PREPROCESS(pil_image).unsqueeze(0)

    cam_engine = gradcam if gradcam is not None else GradCAM(model)

    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    conf = float(probs[pred_idx].item())

    if pred_idx == 1:
        # Generate localization only for FAKE predictions.
        heatmap, _ = cam_engine.compute(input_tensor=input_tensor, target_class=pred_idx)
        overlay = overlay_heatmap_on_rgb(original_rgb=rgb_image, heatmap=heatmap, alpha=0.4)
    else:
        # REAL prediction: no forgery localization map should be produced.
        heatmap = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
        overlay = rgb_image.copy()

    label = "FAKE" if pred_idx == 1 else "REAL"
    return label, conf, overlay, heatmap


def predict_image(image_path, model, gradcam=None):
    """Predict one image path and return label, confidence, and Grad-CAM overlay."""
    image = Image.open(image_path).convert("RGB")
    rgb_image = np.array(image)
    return _predict_with_gradcam_from_rgb(rgb_image=rgb_image, model=model, gradcam=gradcam)


def process_frame_with_localization(frame, heatmap, prediction):
    """
    Draw suspicious regions and heatmap overlay on one RGB frame.

    Args:
        frame: RGB frame (H, W, 3), uint8
        heatmap: float heatmap in [0, 1], shape (H, W)
        prediction: "FAKE" or "REAL"

    Returns:
        annotated_rgb: RGB frame with overlay, boxes, and label text
    """
    annotated = frame.copy()
    frame_h, frame_w = annotated.shape[:2]

    # Normalize heatmap shape/value range, then match frame size for all OpenCV ops.
    heatmap_arr = np.asarray(heatmap)
    if heatmap_arr.ndim == 3:
        heatmap_arr = heatmap_arr[..., 0]
    if heatmap_arr.ndim != 2:
        heatmap_arr = np.zeros((frame_h, frame_w), dtype=np.float32)

    if heatmap_arr.shape != (frame_h, frame_w):
        heatmap_arr = cv2.resize(heatmap_arr, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

    heatmap_arr = np.clip(heatmap_arr, 0.0, 1.0).astype(np.float32)

    # Convert Grad-CAM to 8-bit intensity map expected by OpenCV image ops.
    heatmap_u8 = np.clip(heatmap_arr * 255.0, 0, 255).astype(np.uint8)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    if prediction == "FAKE":
        # Threshold suspicious regions and get contours; handle no detections safely.
        _, binary_mask = cv2.threshold(heatmap_u8, 160, 255, cv2.THRESH_BINARY)
        contour_result = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV 3 returns (image, contours, hierarchy), OpenCV 4 returns (contours, hierarchy).
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        if contours:
            min_area = 40
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        label_text = "FAKE REGION"
        label_color = (255, 64, 64)
    else:
        label_text = "REAL"
        label_color = (64, 255, 64)

    # Semi-transparent heatmap makes localization understandable for non-technical users.
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    annotated = cv2.addWeighted(annotated, 0.72, heatmap_color_rgb, 0.28, 0)

    cv2.putText(
        annotated,
        label_text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        label_color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def generate_annotated_video(frames, output_path, fps=24.0):
    """Reconstruct annotated MP4 video from RGB frames."""
    if not frames:
        return None

    height, width = frames[0].shape[:2]

    # Browser/H.264 encoders usually require even frame dimensions.
    out_width = width if width % 2 == 0 else width - 1
    out_height = height if height % 2 == 0 else height - 1
    out_width = max(out_width, 2)
    out_height = max(out_height, 2)

    codec_candidates = ["avc1", "H264", "mp4v"]
    writer = None
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        test_writer = cv2.VideoWriter(output_path, fourcc, max(float(fps), 1.0), (out_width, out_height))
        if test_writer.isOpened():
            writer = test_writer
            break
        test_writer.release()

    if writer is None or not writer.isOpened():
        return None

    for frame_rgb in frames:
        if frame_rgb.shape[:2] != (out_height, out_width):
            frame_rgb = cv2.resize(frame_rgb, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_path


def predict_video(video_path, model, num_frames=5, gradcam=None, frame_stride=None, output_path=None):
    """
    Predict video frames with Grad-CAM overlays and build annotated output video.

    Returns:
        frame_results: list of tuples(label, confidence, original_rgb, overlay_rgb)
        annotated_video_path: str | None
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return [], None

    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    # Guard against broken metadata that can produce very short output videos.
    fps = input_fps if 1.0 <= input_fps <= 120.0 else 24.0

    stride = max(int(frame_stride or 1), 1)
    preview_indices = set(np.linspace(0, total_frames - 1, max(num_frames, 1), dtype=int).tolist())

    frame_results = []
    processed_frames = []
    last_label = "REAL"
    last_conf = 0.5

    cam_engine = gradcam if gradcam is not None else GradCAM(model)

    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        should_infer = (idx % stride == 0) or (idx in preview_indices)

        if should_infer:
            label, conf, overlay, heatmap = _predict_with_gradcam_from_rgb(
                rgb_image=frame_rgb,
                model=model,
                gradcam=cam_engine,
            )
            last_label = label
            last_conf = conf
        else:
            label = last_label
            conf = last_conf
            overlay = frame_rgb.copy()
            heatmap = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.float32)

        annotated_frame = process_frame_with_localization(
            frame=frame_rgb,
            heatmap=heatmap,
            prediction=label,
        )

        if idx in preview_indices:
            frame_results.append((label, conf, frame_rgb, overlay))
        processed_frames.append(annotated_frame)

    cap.release()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(video_path), "annotated_output.mp4")
    annotated_video_path = generate_annotated_video(
        frames=processed_frames,
        output_path=output_path,
        fps=fps,
    )

    return frame_results, annotated_video_path


def classify_media(path, model, num_video_frames=5, video_frame_stride=None):
    """
    Classify image/video and generate Grad-CAM visualization.

    Image return:
        {"type": "image", "label": str, "confidence": float, "overlay": np.ndarray}

    Video return:
        {
            "type": "video",
            "label": str,
            "confidence": float,
            "annotated_video_path": str | None,
            "frames": [{"label": str, "confidence": float, "original": np.ndarray, "overlay": np.ndarray}, ...]
        }
    """
    mime, _ = mimetypes.guess_type(path)
    media_type = None
    if mime and mime.startswith("image"):
        media_type = "image"
    elif mime and mime.startswith("video"):
        media_type = "video"
    else:
        # Fallback for temp uploads where MIME is missing.
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            media_type = "image"
        elif ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            media_type = "video"

    gradcam = GradCAM(model)

    try:
        if media_type == "image":
            label, conf, overlay, _ = predict_image(path, model, gradcam=gradcam)
            return {
                "type": "image",
                "label": label,
                "confidence": conf,
                "overlay": overlay,
            }

        if media_type == "video":
            frame_results, annotated_video_path = predict_video(
                path,
                model,
                num_frames=num_video_frames,
                gradcam=gradcam,
                frame_stride=video_frame_stride,
            )
            if not frame_results:
                raise ValueError("Could not read frames from video.")

            # Use the first sampled frame as primary prediction summary.
            first_label, first_conf, _, _ = frame_results[0]
            frames = [
                {
                    "label": fr_label,
                    "confidence": fr_conf,
                    "original": fr_orig,
                    "overlay": fr_overlay,
                }
                for fr_label, fr_conf, fr_orig, fr_overlay in frame_results
            ]

            return {
                "type": "video",
                "label": first_label,
                "confidence": first_conf,
                "annotated_video_path": annotated_video_path,
                "frames": frames,
            }

        raise ValueError("Unsupported file type. Use image or video.")
    finally:
        gradcam.remove_hooks()

# Run from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("media_path", help="Path to image/video file")
    parser.add_argument("--output", default="gradcam_overlay.jpg", help="Output path for image overlay")
    parser.add_argument("--video-frames", type=int, default=5, help="Number of sampled video frames")
    parser.add_argument("--video-stride", type=int, default=1, help="Process every Nth frame for video")
    args = parser.parse_args()

    model = load_model()
    result = classify_media(
        args.media_path,
        model,
        num_video_frames=args.video_frames,
        video_frame_stride=max(args.video_stride, 1),
    )

    if result["type"] == "image":
        print(f"\nPrediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, overlay_bgr)
        print(f"Grad-CAM overlay saved to: {args.output}")
    else:
        print(f"\nVideo summary prediction: {result['label']}")
        print(f"Summary confidence: {result['confidence']:.3f}")
        print(f"Annotated video: {result.get('annotated_video_path')}")
        for i, frame in enumerate(result["frames"]):
            out_path = f"gradcam_frame_{i:02d}.jpg"
            overlay_bgr = cv2.cvtColor(frame["overlay"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, overlay_bgr)
            print(f"Frame {i}: {frame['label']} ({frame['confidence']:.3f}) -> {out_path}")

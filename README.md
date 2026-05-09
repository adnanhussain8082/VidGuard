# DeepfakeDetector

DeepfakeDetector is a research-focused deepfake detection project built around EfficientNet-B0, PyTorch Lightning training, Grad-CAM explainability, and a Gradio web interface for image and video analysis.

## What Is In The Project Today

- Binary deepfake classification (REAL vs FAKE) using EfficientNet-B0
- Grad-CAM attention maps for explainability
- Image and video inference paths
- Annotated video generation with highlighted suspicious regions
- PyTorch Lightning training pipeline
- Dataset preparation utilities for frame extraction and train/validation splitting

## Current Tech Stack

- Python
- PyTorch + torchvision
- PyTorch Lightning
- OpenCV
- Gradio
- TensorBoard

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Run

### 1) Web App

```bash
python web-app.py
```

What the web app does:

- Accepts .jpg/.jpeg/.png/.mp4/.mov
- Returns class label + confidence
- Shows original frames and Grad-CAM overlays
- For videos, can output an annotated MP4

### 2) CLI Inference (Image Or Video)

```bash
python classify.py path/to/media
```

Optional CLI arguments:

```bash
python classify.py path/to/video.mp4 --video-frames 5 --video-stride 2 --output gradcam_overlay.jpg
```

Notes:

- Positional argument is media path (not --image)
- For images, output writes an overlay image
- For videos, sampled frame overlays and annotated video are generated

### 3) NPM Script Shortcuts

```bash
npm run dev
npm run train
npm run classify
```

## Training

Start training:

```bash
python main_trainer.py
```

Training pipeline details:

- Reads hyperparameters and dataset paths from config.yaml
- Uses HybridDeepfakeDataset from datasets/hybrid_loader.py
- Model backbone: EfficientNet-B0 with 2-class head
- Saves best checkpoint to models via ModelCheckpoint
- Applies EarlyStopping

Important:

- The current training code uses monitor_metric and log_every_n_steps from config.yaml
- Some scheduler-related keys exist in config.yaml, but scheduler is not currently wired in lightning_modules/detector.py

## Dataset Format

Expected directory layout for each dataset path:

```text
dataset_root/
  real/
    img1.jpg
    ...
  fake/
    img1.jpg
    ...
```

In config.yaml, train_paths and val_paths are lists of roots that contain real/ and fake/ subfolders.

## Explainability And Video Behavior

Grad-CAM implementation lives in gradcam_utils.py.

Inference behavior in classify.py:

- REAL prediction: no suspicious-region localization heatmap is produced
- FAKE prediction: Grad-CAM heatmap is generated and overlaid
- Video mode:
  - Supports stride-based frame processing
  - Reuses latest prediction on skipped frames
  - Builds annotated_output.mp4 with overlay + optional suspicious boxes

## Export And Utility Scripts

- inference/export_onnx.py: exports model to deepfake_model.onnx
- inference/video_inference.py: folder-based video prediction script (expects videos_to_predict)
- tools/split_dataset.py: image dataset split helper
- tools/split_train_val.py: extracts frames from videos into dataset folders
- tools/split_video_dataset.py: split + frame extraction pipeline
- tools/export_to_pt.py: Lightning checkpoint to PyTorch state dict export

## Model File Path Notes

Different scripts currently reference different model filenames:

- classify.py and web-app.py default to models/best_model-v3.pt
- realeval.py loads models/best_model-v2.pt
- inference/export_onnx.py and inference/video_inference.py load models/best_model.pt

Before running every script, align model filenames or update script paths to your actual checkpoint/state-dict file.

## Project Structure

```text
DeepfakeDetector/
  README.md
  PRESENTATION.md
  LICENSE
  requirements.txt
  package.json
  config.yaml
  main_trainer.py
  classify.py
  web-app.py
  realeval.py
  gradcam_utils.py
  datasets/
    hybrid_loader.py
  lightning_modules/
    detector.py
  inference/
    export_onnx.py
    video_inference.py
  tools/
    export_to_pt.py
    split_dataset.py
    split_train_val.py
    split_video_dataset.py
  models/
    best_model-v3.pt
```

## Known Limitations

- Binary classifier only (no manipulation-type classification)
- No temporal deep model (video is frame-based inference)
- Metrics in training module are basic (loss and accuracy)
- Some utility scripts contain example hardcoded paths that should be adjusted

## Recommended Next Cleanup

- Unify model artifact naming across all scripts
- Add argparse to utility scripts that currently rely on in-file constants
- Add richer evaluation metrics (precision/recall/F1/AUC/confusion matrix)
- Add augmentation and scheduler support directly in training module

## License

MIT License

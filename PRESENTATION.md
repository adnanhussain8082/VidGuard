# DeepfakeDetector Project Presentation

## Updated Slide Deck (Aligned With Current Codebase)

---

## SLIDE 1: Title

### DeepfakeDetector

### Explainable Deepfake Detection For Images And Videos

- Built with EfficientNet-B0, PyTorch Lightning, OpenCV, and Gradio
- Includes Grad-CAM explainability and annotated video output
- Academic pre-project with practical demo interface

Speaker notes:

- Open with why deepfake detection matters for media trust.
- Position this as a working engineering prototype, not just a concept.

---

## SLIDE 2: Problem And Objective

### Problem

- Deepfakes are increasingly realistic and hard to detect visually
- Non-technical users often lack accessible forensic tools
- Manual verification is slow and inconsistent

### Objective

- Build a usable detector for REAL vs FAKE media
- Provide visual explanation (Grad-CAM)
- Support both command line workflows and web-based demo usage

Speaker notes:

- Mention journalist, moderation, and research use cases.
- Emphasize explainability as a trust layer, not just prediction output.

---

## SLIDE 3: System Architecture

### Core Design

- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Head: Dropout(0.4) + Linear(2) for binary class output
- Input: 224x224 RGB + ImageNet normalization
- Output: softmax probabilities for REAL and FAKE

### Key Modules

- main_trainer.py: training orchestration
- lightning_modules/detector.py: Lightning module
- classify.py: media classification + Grad-CAM flow
- gradcam_utils.py: attention map generation and overlay

Speaker notes:

- Explain why EfficientNet-B0 was chosen: good balance of accuracy and efficiency.

---

## SLIDE 4: Training Pipeline (Current Implementation)

### How Training Works

- Config loaded from config.yaml
- Dataset loader: HybridDeepfakeDataset
- Trainer callbacks:
  - ModelCheckpoint (best model by monitor metric)
  - EarlyStopping
- Optimizer: Adam
- Loss: CrossEntropyLoss

### Metrics Logged

- train_loss, train_acc
- val_loss, val_acc

### Important Reality Check

- config.yaml contains scheduler settings
- current detector.py does not yet apply LR scheduler in configure_optimizers

Speaker notes:

- This slide should reflect actual code behavior, not planned behavior.

---

## SLIDE 5: Data Handling And Supported Format

### Expected Dataset Structure

- Each dataset root has:
  - real/
  - fake/
- Train and validation are provided as lists in config.yaml

### Supported File Types In App

- Images: .jpg, .jpeg, .png
- Videos: .mp4, .mov (web app), broader support in code path fallback

### Utilities

- tools/split_dataset.py
- tools/split_train_val.py
- tools/split_video_dataset.py

Speaker notes:

- Clarify that utility scripts include example hardcoded paths and may need editing before use.

---

## SLIDE 6: Inference Flow

### classify.py Behavior

- Accepts one media path (image or video)
- Image path:
  - Predict REAL/FAKE
  - Generate Grad-CAM only for FAKE prediction
- Video path:
  - Frame sampling + optional stride-based processing
  - Annotated video reconstruction

### CLI Example

```bash
python classify.py path/to/media --video-frames 5 --video-stride 2
```

Speaker notes:

- Point out that this is positional media_path, not a --image flag.

---

## SLIDE 7: Grad-CAM Explainability

### What Is Implemented

- Automatic last Conv2d layer detection
- Hook-based activation/gradient capture
- Heatmap normalization and RGB overlay
- Suspicious region highlighting for FAKE outputs

### Why It Matters

- Makes predictions interpretable
- Helps users inspect suspicious regions visually
- Improves trust and debugging capability

Speaker notes:

- Show one REAL and one FAKE output to compare attention behavior.

---

## SLIDE 8: Web Application

### Gradio Interface Features

- Drag-and-drop media upload
- REAL/FAKE label + confidence
- Original frame gallery
- Grad-CAM overlay gallery
- Annotated video player output
- Video frame stride control

### UX Direction

- Enterprise-style dark UI with custom CSS
- Local processing workflow

Speaker notes:

- Demonstrate one image and one short video if possible.

---

## SLIDE 9: Export And Deployment Utilities

### Available Export Paths

- inference/export_onnx.py
  - Exports to deepfake_model.onnx
- tools/export_to_pt.py
  - Converts Lightning checkpoint to PyTorch state dict

### Additional Script

- inference/video_inference.py
  - Batch-style folder inference (videos_to_predict)

Speaker notes:

- Mention these scripts are useful but currently not fully parameterized by CLI args.

---

## SLIDE 10: Current Gaps And Risks

### Practical Gaps Identified

- Model path inconsistency across scripts:
  - best_model-v3.pt, best_model-v2.pt, best_model.pt
- Utility scripts rely on example hardcoded paths
- Evaluation metrics are basic (no precision/recall/F1/AUC)
- No temporal deep architecture for videos yet
- No integrated scheduler although config has scheduler keys

Speaker notes:

- This honesty improves credibility and helps justify next-phase work.

---

## SLIDE 11: Next Iteration Plan

### Engineering Priorities

1. Unify model artifact naming and loading strategy
2. Add argparse to all utility scripts
3. Add richer evaluation suite (PR, RC, F1, AUC, confusion matrix)
4. Integrate scheduler and augmentation into training pipeline
5. Add temporal modeling option for video consistency
6. Add benchmark table across datasets

Speaker notes:

- Keep this as concrete and executable roadmap items.

---

## SLIDE 12: Conclusion

### What Has Been Achieved

- End-to-end deepfake detector is functional
- Explainability is integrated via Grad-CAM
- Web app and CLI both available
- Annotated video output improves practical usability

### Final Message

- The project is a solid base for 8th semester expansion into stronger evaluation and more robust video modeling.

---

## Appendix: Demo Commands

```bash
pip install -r requirements.txt
python main_trainer.py
python web-app.py
python classify.py path/to/sample.jpg
python classify.py path/to/sample.mp4 --video-frames 5 --video-stride 2
```

---

## Appendix: Key Files

- main_trainer.py
- classify.py
- web-app.py
- gradcam_utils.py
- lightning_modules/detector.py
- datasets/hybrid_loader.py
- inference/export_onnx.py
- inference/video_inference.py
- tools/split_dataset.py
- tools/split_train_val.py
- tools/split_video_dataset.py

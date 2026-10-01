# PPE Detection with YOLOv8

A computer vision project by **Rhea Malpekar** that detects people and personal protective equipment in images, video, and a local webcam feed. The model fine-tunes YOLOv8n on a construction-worker dataset using a GPU runtime in Google Colab.

## Supported classes

| Class ID | Label | Description |
|---|---|---|
| 0 | gloves | Protective work gloves |
| 1 | helmet | Safety helmets / hard hats |
| 2 | human | People / workers |
| 3 | vest | Safety vests |

Boots annotations were removed from the original dataset and the remaining class IDs were remapped before training. Detecting PPE objects does not automatically establish which person is wearing each item or certify safety compliance.

## Sample demo images

These **AI-generated synthetic input photos** are supplied for demonstration. They are not dataset samples, annotated ground truth, or model predictions. No boxes or confidence scores have been added. Model outputs may differ from the visible contents.

### Worker wearing PPE

![Synthetic worker wearing helmet, vest, and gloves](assets/demo/full_ppe.png)

Use this image to demonstrate detection of a person and visible PPE.

### Workers with mixed PPE

![Synthetic workers with different visible PPE](assets/demo/mixed_ppe.png)

Use this image to inspect detections in a scene with multiple people and different visible equipment. A missed detection alone does not prove an item is absent.

## Training setup

| Setting | Configuration |
|---|---|
| Model | YOLOv8n, initialized from pretrained weights |
| Training environment | Google Colab, NVIDIA Tesla T4 GPU |
| Training images | 4,368 |
| Validation images | 420 |
| Maximum epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Early stopping patience | 10 epochs |
| Dataset source | Roboflow: `lexiwave-glqpu / construction-worker-9yqmx`, version 1 |

The values above are documented in the supplied notebook PDF. Keep the source dataset's attribution and license requirements when redistributing dataset images or weights.

## Evaluation

The earlier project discussion reported approximate validation results of **90% mAP@0.50** and **71% mAP@0.50:0.95** from completed training curves. Those completed results and model weights are not included in this demo package; the supplied notebook PDF ends before final evaluation output. Confirm the figures against `results.csv` or a new validation run before presenting them as verified results.

- **mAP@0.50:** average precision across classes at an IoU threshold of 0.50.
- **mAP@0.50:0.95:** average precision across classes and IoU thresholds from 0.50 to 0.95.
- Review per-class results, precision, recall, and failure cases alongside overall mAP.

To evaluate your trained checkpoint using your cleaned dataset configuration:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
metrics = model.val(data="path/to/data_no_boots.yaml", imgsz=640)
print("mAP@0.50:", metrics.box.map50)
print("mAP@0.50:0.95:", metrics.box.map)
print("Per-class mAP@0.50:0.95:", metrics.box.maps)
```

`metrics.box.maps` contains per-class mAP@0.50:0.95; the original notebook's “Per-class mAP50” print label should be corrected.

## Quick start

1. Extract this package into your project root, keeping the image paths intact.
2. Place your PPE-trained `best.pt` in the same directory as this README. The weights are **not included**. Generic `yolov8n.pt` weights do not represent the trained four-class PPE model.
3. Create a Python environment and install Ultralytics:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ultralytics
```

On Windows, activate with `.venv\Scripts\activate` instead.

### Predict on the sample images

Run from the project root:

```bash
yolo predict model=best.pt source=assets/demo imgsz=640 conf=0.25 save=True project=runs/demo name=samples
```

The command saves genuine prediction images under the output directory printed by Ultralytics, normally `runs/demo/samples`. Subsequent runs may create a numbered directory. The confidence threshold is a starting point for the demo, not a calibrated safety threshold.

### Webcam demo

Run this locally on a computer with a webcam:

```bash
yolo predict model=best.pt source=0 imgsz=640 conf=0.25 show=True
```

A Colab runtime cannot directly use your laptop's webcam through `source=0`. Use uploaded images for the Colab demo, or run webcam inference locally.

### Video demo

```bash
yolo predict model=best.pt source=path/to/video.mp4 imgsz=640 conf=0.25 save=True
```

## Add real model outputs to this README

After running image inference, copy the saved prediction images into `assets/results/`. Then add these lines with filenames that match your actual files:

```markdown
![PPE detections on one worker](assets/results/full_ppe.png)
![PPE detections on two workers](assets/results/mixed_ppe.png)
```

Only use images produced by your trained model for the prediction section. Keep the synthetic input captions above so readers can distinguish input photos from measured outputs.

## Demo package contents

- `README.md` — project overview, training configuration, evaluation notes, and demo commands.
- `assets/demo/full_ppe.png` — synthetic single-worker input photo.
- `assets/demo/mixed_ppe.png` — synthetic two-worker input photo.
- `assets/demo/PROMPTS.md` — generation prompts and image provenance.

Add your actual notebook, trained weights, and evaluation artifacts to your repository separately.

## Limitations

Small gloves, occlusion, low light, camera distance, and unfamiliar scenes can affect performance. Synthetic images are useful for demonstrations but do not replace evaluation on real, held-out images. This project demonstrates object detection and requires human review for safety decisions.

## Acknowledgments

Built with Ultralytics YOLOv8, PyTorch, Google Colab, and a Roboflow construction-worker dataset. The sample input photos were created with AI image generation for this README; they contain no measured model results.

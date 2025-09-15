# Shuttlecock Small Object Detection Experiment: YOLOv11 vs YOLOv11+SAHI
## Project Overview

This project implements small object detection for shuttlecocks using YOLOv11 and SAHI. We conducted a comparative study between YOLOv11 native detection and YOLOv11 + SAHI sliced prediction.

## Main Objectives

Train a custom YOLOv11 model to detect shuttlecocks

Improve small object detection accuracy using SAHI sliced prediction

Compare the performance of both methods on small object detection

## Dataset

The dataset includes training and testing sets.Due to limited dataset size, the training set is also used as the validation set. As a result, the evaluation during training reflects performance on the same data used for model optimization. While this setup is sufficient for demonstrating the concept and experimental workflow, it may overestimate the model’s generalization ability.

Only one class exists: badminton, with class_id = 0

YOLO format is used for labels:

<class_id> <x_center> <y_center> <width> <height>

Coordinates are normalized to [0,1]

Future work will include a separate validation set to more accurately assess performance on unseen data and to prevent overfitting.

YOLOv11 Model Training
```python
from ultralytics import YOLO

# Create a new YOLOv11 model from scratch
model = YOLO("myyolo11n.yaml")

# Load a pretrained YOLOv11 model (recommended)
model = YOLO("E:/VisualProject/ultralytics-main/yolo11n.pt")

# Train the model using the custom dataset for 30 epochs
results = model.train(
    data="E:/VisualProject/ultralytics-main/ultralytics/cfg/datasets/mycoco128.yaml",
    epochs=30
)

# Evaluate the model on the validation set
results = model.val()

# (Optional) Perform detection on a single image
# results = model("https://ultralytics.com/images/bus.jpg")

# (Optional) Export the model to ONNX format
# success = model.export(format="onnx")
```

Experiment 1: YOLOv11 Native Detection
```python
from ultralytics import YOLO

model = YOLO("E:/VisualProject/ultralytics-main/runs/detect/train2/weights/best.pt")

# Predict on all images in the test folder
results = model.predict(
    source="E:/VisualProject/ultralytics-main/data/mycoco/test_images",
    show=True,
    save=True
)
```

### Note: Native YOLOv11 may miss small objects such as shuttlecocks.

Experiment 2: YOLOv11 + SAHI Sliced Prediction
```python
import os
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

# 1. Configure YOLOv11 model
model = UltralyticsDetectionModel(
    model_path="E:/VisualProject/ultralytics-main/runs/detect/train2/weights/best.pt",
    confidence_threshold=0.3,
    device="cpu"  # Change to "cuda:0" if GPU is available
)

# 2. Configure test image folder and output folder
image_folder = "E:/VisualProject/ultralytics-main/data/mycoco/test_images"
output_folder = "E:/VisualProject/ultralytics-main/sahi_results"
os.makedirs(output_folder, exist_ok=True)

# 3. Process all jpg images in the folder
for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(".jpg"):
        img_path = os.path.join(image_folder, img_name)
        print(f"Processing: {img_name}")

        # SAHI sliced prediction
        result = get_sliced_prediction(
            img_path,
            detection_model=model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        # 4. Save visualization results
        result.export_visuals(
            export_dir=output_folder,
            file_name=f"{os.path.splitext(img_name)[0]}_pred.jpg"
        )
        result.export_json(
            export_dir=output_folder,
            file_name=f"{os.path.splitext(img_name)[0]}.json"
        )
```

### SAHI sliced prediction significantly improves detection of small objects, especially shuttlecocks.

## Comparative Results
### YOLOV11
![262174269932568617](https://github.com/user-attachments/assets/4e110cd6-56a2-453b-8ec0-957dea0caf0a)
![t014cfe36fe7f82b925](https://github.com/user-attachments/assets/2b95f6e3-fae3-4550-b237-1b1dfc37ca91)
![tqjia](https://github.com/user-attachments/assets/c9e74992-72d2-4087-847b-751d60376d5f)
![下载](https://github.com/user-attachments/assets/7570c545-3f40-4da9-b832-e23856da056c)
### YOLOV11+SAHI
<img width="1023" height="682" alt="262174269932568617_pred jpg" src="https://github.com/user-attachments/assets/a3f1aa3c-2306-4d2d-b93b-155b677eb96c" />
<img width="251" height="400" alt="t014cfe36fe7f82b925_pred jpg" src="https://github.com/user-attachments/assets/1eb6c5eb-e0dd-4b56-953f-9091e660d370" />
<img width="700" height="400" alt="tqjia_pred jpg" src="https://github.com/user-attachments/assets/a59306c1-b2a6-4743-a779-9293733fe2ac" />
<img width="319" height="234" alt="下载_pred jpg" src="https://github.com/user-attachments/assets/6280ed47-e07f-46c7-98fc-7a02152e81ba" />

## Limitations:

### Small object missed detection: Although SAHI improves detection, very tiny or heavily occluded shuttlecocks may still be missed.

### Processing time: Sliced prediction increases inference time due to overlapping slices and multiple forward passes.

### Single-class detection: Current model is trained only for shuttlecocks; generalization to other object types is not evaluated.

### Limited dataset size: The dataset may be relatively small, potentially affecting model robustness in diverse scenarios.

Future Improvements:

### Data augmentation: Apply advanced augmentation techniques (e.g., mosaic, mixup, random cropping) to increase small object diversity.

### Hyperparameter tuning: Experiment with different confidence thresholds, slice sizes, and overlap ratios to balance accuracy and speed.

### Multi-class expansion: Extend the model to detect multiple small object types simultaneously.

### Model optimization: Consider lighter-weight architectures or GPU acceleration to reduce inference time.

### Ensemble methods: Combine YOLOv11 + SAHI predictions with other detection models to further improve recall on challenging small objects.







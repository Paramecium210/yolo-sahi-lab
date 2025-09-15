羽毛球小物体检测实验：YOLOv8 与 YOLOv8+SAHI 对比
项目简介

本项目基于 YOLOv8
 与 SAHI
 实现羽毛球小物体检测，并进行了 YOLOv8 原生检测 vs YOLOv8 + SAHI 切片推理 的对比实验。

主要研究内容：

自定义 YOLOv8 模型训练，识别羽毛球

SAHI 切片推理提高小物体检测精度

对比两种方法在小物体检测上的性能差异

数据集

数据集包括训练集、验证集和测试集

目录结构：

dataset/
├─ images/
│  ├─ train/
│  ├─ val/
│  ├─ test/
├─ labels/
│  ├─ train/
│  ├─ val/


标签采用 YOLO 格式：

<class_id> <x_center> <y_center> <width> <height>


坐标归一化到 [0,1]

class_id 从 0 开始，只有“羽毛球”这一类

环境依赖
pip install ultralytics sahi scikit-image imagecodecs


Python >= 3.8

GPU 可选（CUDA）

YOLOv8 模型训练
from ultralytics import YOLO

# 加载预训练权重
model = YOLO("yolov8n.pt")  

# 开始训练
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="0"  # CPU 或 GPU
)


输出权重文件路径：runs/detect/train/weights/best.pt

实验 1：YOLOv8 原生检测
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="dataset/images/test",
    imgsz=640,
    conf=0.3,
    save=True,
    save_txt=True,
)


原生 YOLOv8 对小物体可能存在漏检

实验 2：YOLOv8 + SAHI 切片推理
import os
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

model = UltralyticsDetectionModel(
    model_path="runs/detect/train/weights/best.pt",
    confidence_threshold=0.3,
    device="cpu"
)

image_folder = "dataset/images/test"
output_folder = "sahi_results"
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(".jpg"):
        img_path = os.path.join(image_folder, img_name)
        result = get_sliced_prediction(
            img_path,
            detection_model=model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        base_name = os.path.splitext(img_name)[0]
        result.export_visuals(export_dir=output_folder, file_name=f"{base_name}_pred.jpg")
        result.export_json(export_dir=output_folder, file_name=f"{base_name}.json")


SAHI 切片推理显著提升小物体检测率，尤其是羽毛球这种尺寸较小的目标

每张图片会生成带检测框的可视化图和对应 JSON 文件

对比实验效果
方法	检测精度	小物体漏检	可视化示例
YOLOv8 原生检测	较低	存在	保存于 runs/detect/pred
YOLOv8 + SAHI	较高	极少	保存于 sahi_results/

可在 README 中插入 原生 YOLO 与 SAHI 对比图，直观展示改进效果

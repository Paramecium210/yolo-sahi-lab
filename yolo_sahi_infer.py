import os
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction


# 1 配置 YOLOv8 模型

model = UltralyticsDetectionModel(
    model_path="E:/VisualProject/ultralytics-main/runs/detect/train2/weights/best.pt",  # 替换成你的训练权重
    confidence_threshold=0.3,
    device="cpu"  # 如果有 GPU 改成 "cuda:0"
)


# 2 配置测试图片路径和输出路径
image_folder = "E:/VisualProject/ultralytics-main/data/mycoco/test_images"  # 测试图片文件夹
output_folder = "E:/VisualProject/ultralytics-main/sahi_results"           # 输出文件夹



# 3 遍历文件夹里的所有 jpg 图片
for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(".jpg"):
        img_path = os.path.join(image_folder, img_name)
        print(f"正在处理: {img_name}")

        # SAHI 切片推理
        result = get_sliced_prediction(
            img_path,
            detection_model=model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )


        # 4 保存可视化结果
        result.export_visuals(
            export_dir=output_folder,
            file_name=f"{os.path.splitext(img_name)[0]}_pred.jpg"
        )
print("所有图片检测完成！")

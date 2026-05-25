from ultralytics import YOLO

model = YOLO("E:/VisualProject/ultralytics-main/runs\detect/train2\weights/best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")摄像头检测
results = model.predict(
    source="E:/VisualProject/ultralytics-main/data/mycoco/test_images", show=True, save=True
)  # Display preds. Accepts all YOLO predict arguments

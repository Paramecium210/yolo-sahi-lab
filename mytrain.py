from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("myyolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("E:/VisualProject/ultralytics-main/yolo11n .pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="E:/VisualProject/ultralytics-main/ultralytics/cfg/datasets/mycoco128.yaml", epochs=30)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
#success = model.export(format="onnx")
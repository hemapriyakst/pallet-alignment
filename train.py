from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # start from pretrained weights

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="pallet_model",
    patience=10
)

print("Training complete! Weights saved in runs/detect/pallet_model/weights/")
from ultralytics import YOLO

model = YOLO("models/best.pt")
print(model.names)
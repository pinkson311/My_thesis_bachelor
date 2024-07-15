from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")
if __name__ == '__main__':
    # Train the model
    results = model.train(data='data.yaml', name="detect-v8m", epochs=1000, imgsz=640, batch=4)
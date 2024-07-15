import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO

# Define the DistanceModel class (same as during training)
class DistanceModel(nn.Module):
    def __init__(self):
        super(DistanceModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO(r'yolov8\runs\detect\detect-v8m\weights\best.pt')

distance_model = DistanceModel().to(device)
distance_model.load_state_dict(torch.load('best_model.pth', map_location=device))
distance_model.eval()

def get_bounding_boxes(image):
    # Phát hiện bounding box từ hình ảnh bằng mô hình YOLOv8
    results = yolo_model(image)
    boxes = []
    for result in results:
        for bbox in result.boxes.xyxy:
            boxes.append(bbox.cpu().numpy())
    return boxes

def predict_distance(bounding_boxes):
    distances = []
    for box in bounding_boxes:
        coords = np.array(box[:4], dtype=np.float32)  # lấy tọa độ bounding box
        coords_tensor = torch.tensor(coords).to(device).view(1, -1)
        with torch.no_grad():
            distance = distance_model(coords_tensor).item()
        distances.append(distance)
    return distances

def draw_boxes(image, bounding_boxes, distances):
    for (box, distance) in zip(bounding_boxes, distances):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Distance: {distance:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

# Đường dẫn tới ảnh đầu vào
image_path = r"C:\Users\Admin\Desktop\xetest.jpg"
image = cv2.imread(image_path)

# Lấy bounding boxes từ ảnh
bounding_boxes = get_bounding_boxes(image)

# Dự đoán khoảng cách từ bounding boxes
distances = predict_distance(bounding_boxes)

# Vẽ bounding boxes và khoảng cách lên ảnh
output_image = draw_boxes(image, bounding_boxes, distances)

# Hiển thị ảnh kết quả
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)  # Đợi người dùng nhấn phím bất kỳ để đóng cửa sổ hiển thị
cv2.destroyAllWindows()


# Lưu ảnh kết quả
output_image_path = 'output_image.png'
cv2.imwrite(output_image_path, output_image)

print("Inference completed. Output image saved.")
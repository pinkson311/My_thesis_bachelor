import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def preprocess_image(image_path, target_size=(640, 640)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img

# Hàm lấy tọa độ bounding box từ YOLOv8
def get_bbox_from_yolo(image):
    results = yolo_model(image)
    bboxes = results.xyxy[0].cpu().numpy()  # Lấy kết quả bounding box
    return bboxes

# Hàm inference
def inference(image_path):
    # Tiền xử lý ảnh
    device = 'cuda'
    preprocessed_img = preprocess_image(image_path)
    preprocessed_img = preprocessed_img.to(device)
    
    # Lấy bounding box từ YOLOv8
    original_img = cv2.imread(image_path)
    bboxes = get_bbox_from_yolo(original_img)
    
    # Chọn bounding box đầu tiên (giả sử chỉ có một vật thể cần đo khoảng cách)
    if len(bboxes) > 0:
        bbox = bboxes[0]
        xmin, ymin, xmax, ymax = bbox[:4]
        
        # Tính toán các tọa độ trung tâm và kích thước bounding box
        img_height, img_width = original_img.shape[:2]
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Tạo đầu vào cho mô hình neural network
        input_coords = torch.tensor([[x_center, y_center, width, height]], dtype=torch.float32).to(device)

        # Dự đoán khoảng cách bằng mô hình neural network
        with torch.no_grad():
            distance = model(preprocessed_img, input_coords)
        
        return distance.item()
    else:
        return None

# Load the dataframe
df = pd.read_csv("annotations.csv")

# Group the annotations by filename
grouped = df.groupby('filename')

# Iterate over each group (each image)
for filename, group in grouped:
    if filename == "original_data/train_images/000050.png":
        # Read the image
        im = cv2.imread(filename)
        
        # Draw the center line
        
        # Iterate over all rows in the group (all annotations for this image)
        for idx, row in group.iterrows():
            x1 = int(row['x_min'])
            y1 = int(row['y_min'])
            x2 = int(row['x_max'])
            y2 = int(row['y_max'])

            # Draw the bounding box
            im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw the class and distance text
            string = "({})".format(row['class'])
            # string = "({} {:.2f})".format(row['class'], row['distance'])
            im = cv2.putText(im, string, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display the image with annotations
        cv2.imwrite("detections.png", im)
        cv2.imshow("detections", im)
        cv2.waitKey(0)

cv2.destroyAllWindows()

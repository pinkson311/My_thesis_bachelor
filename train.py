import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BoundingBoxDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        coords = self.annotations.iloc[idx, 2:6].to_numpy().astype(np.float32)
        distance = self.annotations.iloc[idx, 6].astype(np.float32)

        return coords, distance

# Đường dẫn đến dữ liệu
annotations_file = 'annotations.csv'

# Tạo DataLoader cho tập huấn luyện và tập kiểm tra
dataset = BoundingBoxDataset(annotations_file=annotations_file)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


import torch.nn as nn

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistanceModel().to(device)


import torch.optim as optim
from tqdm import tqdm

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50000
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for coords, distances in tqdm(train_loader):
        coords = coords.to(device)
        distances = distances.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(coords)
        loss = criterion(outputs, distances)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Lưu mô hình nếu loss nhỏ nhất
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved model with loss {best_loss:.4f}")
        print("============================================")


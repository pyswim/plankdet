'''
data/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ... (共8个文件夹)
└── val/
    ├── class_0/
    ├── class_1/
    └── ...
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os

# 超参数设置
BATCH_SIZE = 32
EPOCHS = 50#10
LEARNING_RATE = 0.001#0.001
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 数据集路径（请修改为实际路径）
train_dir = 'datasets/train'
val_dir = 'datasets/val'

# 加载原始数据集（包含全部图片）
train_dataset_full = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset_full = datasets.ImageFolder(root=val_dir, transform=transform)

# 获取类别名称（直接从原始数据集获取）
class_names = train_dataset_full.classes
print("类别:", class_names)

# 定义函数：限制每个类别的样本数（返回 Subset）
def limit_samples_per_class(dataset, max_samples=100, random_seed=42):
    """
    从 dataset 中为每个类别随机选择最多 max_samples 个样本，
    返回 Subset 对象。
    """
    targets = np.array(dataset.targets)          # 所有样本的标签
    classes = np.unique(targets)                  # 类别ID列表
    selected_indices = []

    np.random.seed(random_seed)

    for class_id in classes:
        class_indices = np.where(targets == class_id)[0]
        if len(class_indices) > max_samples:
            chosen = np.random.choice(class_indices, size=max_samples, replace=False)
        else:
            chosen = class_indices
        selected_indices.extend(chosen)

    np.random.shuffle(selected_indices)
    return Subset(dataset, selected_indices)

# 限制训练集和验证集每个类别最多100张图片
train_dataset = limit_samples_per_class(train_dataset_full, max_samples=100)
val_dataset = limit_samples_per_class(val_dataset_full, max_samples=100)   # 验证集也可限制

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------- 模型定义 --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),#16*128*128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),#16*64*64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),#32*64*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),#32*32*32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),#64*32*32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),#64*16*16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#128*16*16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),#128*8*8
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),#nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)# 64),#num_classes),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            #nn.Linear(64, 32),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            #nn.Linear(32, 16),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            #nn.Linear(16, num_classes),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------- 训练与验证函数 --------------------
def train_one_epoch(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -------------------- 训练循环 --------------------

test_ds=datasets.ImageFolder('datasets/test',transform=transform)
test_loader=DataLoader(test_ds,batch_size=1)
ts=[i for i in test_loader]


best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer)
    val_loss, val_acc = validate(val_loader, model, criterion)
    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  -> Best model saved (acc={val_acc:.4f})")

print("训练完成！")

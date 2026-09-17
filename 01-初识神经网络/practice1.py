# Day1 + Day2 的代码片段练习（修正版）
# 分开两个文件写，这里放一起方便复习对照

# ============================================================
# 通用导入
# ============================================================
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Day 1: moons 二分类
# ============================================================

# ---- 数据 ----
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.1)
X = torch.tensor(X).float()
y = torch.tensor(y).float()
y = y * 2 - 1                                  # 0/1 → -1/1

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---- 模型 ----
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        sz = layer_sizes
        self.layers = nn.ModuleList([
            nn.Linear(sz[i], sz[i+1]) for i in range(len(sz) - 1)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):       # ← 名字统一用 layer
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

# ---- 训练 ----
random.seed(1337)                                    # ← seed 是函数
np.random.seed(1337)
torch.manual_seed(1337)

model = MLP([2, 16, 16, 1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    loss_epoch, correct, total = 0, 0, 0
    for Xb, yb in dataloader:
        scores = model(Xb)
        loss = (1 - yb.unsqueeze(1) * scores).clamp(min=0).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        correct += ((scores > 0) == (yb.unsqueeze(1) > 0)).sum().item()
        total += len(yb)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss_epoch:.3f}, acc={correct/total*100:.1f}%")


# ============================================================
# Day 2: MNIST 多分类
# ============================================================

from torchvision import datasets
import torchvision.transforms as T

# ---- 数据 ----
transform = T.ToTensor()                                       # ← 拼写对

train_data = datasets.MNIST(root='./data', train=True,
                            download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False,
                            download=True, transform=transform)

train_set, val_set = torch.utils.data.random_split(train_data, [0.8, 0.2])
#                       ↑ random_split 不是 random.split

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
#                                                     ↑ 测试集不 shuffle

# ---- 模型（复用同一个 MLP 类，换参数）----
model = MLP([784, 256, 256, 10])

# ---- 训练 ----
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    loss_train = 0
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    loss_val = 0
    for images, labels in val_loader:
        with torch.no_grad():
            loss = loss_fn(model(images), labels)
            loss_val += loss.item()

    print(f"Epoch {epoch}: train={loss_train/len(train_loader):.4f}, "
          f"val={loss_val/len(val_loader):.4f}")

# ---- 测试 ----
correct, total = 0, 0
for images, labels in test_loader:
    with torch.no_grad():                                     # ← 测试也要 no_grad
        preds = model(images)
        _, predicted = torch.max(preds, 1)                    # ← dim=1 不能省
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试准确率: {correct / total * 100:.2f}%")

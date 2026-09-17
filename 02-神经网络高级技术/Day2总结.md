# Day2 总结：MNIST 多分类 & 过拟合应对

---

## 一、任务变化

| | Day 1 | Day 2 |
|---|---|---|
| 数据 | 100 个月牙坐标 | 60000 张手写数字图（28×28） |
| 输入维度 | 2 | 784（= 28×28） |
| 类别数 | 2（±1） | 10（0~9） |
| 损失函数 | Hinge Loss | CrossEntropyLoss |
| 标签格式 | ±1 | 整数 0~9 |
| 评估指标 | `score > 0` 判断正负 | `torch.max(preds, 1)` 取最大类 |
| 数据集划分 | 全量训练 | 训练集/验证集/测试集 |

---

## 二、PyTorch 数据体系：Dataset / DataLoader / random_split

### 一张图搞定

```
原始数据 (NumPy / 图片文件 / 任何东西)
    │
    ▼  包装成 Dataset
Dataset  —  定义了 "怎么取第 i 条数据"
    │
    ▼  套上 DataLoader
DataLoader  —  定义了 "怎么分 batch、怎么打乱、怎么并行加载"
```

### Dataset：数据容器（三种方式创建）

```python
# 方式 1：手写数据直接打包（Day 1）
dataset = TensorDataset(X, y)          # dataset[i] → (x, y)

# 方式 2：torchvision 自带标准数据集（Day 2）
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=...)
                                       # dataset[i] → (图片, 标签)

# 方式 3：手写 Dataset 类（自己的非标准数据）
class MyDataset(Dataset):
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.label[i]
```

### DataLoader：自动分 batch + 打乱

```python
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#              ↑                  ↑                ↑
#           套在 Dataset 上     每批 64 条      每个 epoch 随机打乱
```

DataLoader 做的事：

```
dataset: [0, 1, 2, 3, 4, ..., 99]           ← 100 条
shuffle 后: [23, 81, 5, 67, ...]            ← 随机重排
分 batch (size=64):
  batch 1: [23, 81, 5, ..., 44]  ← 64 条
  batch 2: [9, 33, 72, ..., 9]   ← 36 条
```

### random_split：把 Dataset 切成两份（Day 2 用到）

```python
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
#                    ↑ 函数，不是类
#                    60000 条按 8:2 随机切成两份
```

### 完整关系

```
                    TensorDataset(X, y)      ← Day 1
                    datasets.MNIST(...)      ← Day 2
                    MyDataset(...)           ← 自定义
                         │
                         │  都是 Dataset，实现 dataset[i]
                         ▼
              ┌──────────────────┐
              │     Dataset      │
              └────────┬─────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
    random_split   DataLoader   DataLoader
          │        (train)       (test, shuffle=False)
          ▼         │            │
    train_set       ▼            ▼
    val_set       batch          batch
                  shuffle        no shuffle
```

---

## 三、核心代码范式（加上了验证集 + 测试）

### 1. 加载图像数据

```python
from torchvision import datasets
import torchvision.transforms as T

transform = T.ToTensor()                           # 图片 → Tensor + 像素归一化 0~1

train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

### 2. 划分训练/验证 + DataLoader

```python
from torch.utils.data import DataLoader

train_set, val_set = torch.utils.data.random_split(train_data, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)    # 测试集不打乱
```

### 3. 模型（和 Day1 同模板，只改了输入输出维度）

```python
class MLP(nn.Module):
    def __init__(self, layer_sizes):                # [784, 256, 256, 10]
        super().__init__()
        sz = layer_sizes
        self.layers = nn.ModuleList([
            nn.Linear(sz[i], sz[i+1])
            for i in range(len(sz) - 1)
        ])

    def forward(self, x):
        x = x.view(-1, 28*28)                       # (B,1,28,28) → (B,784) 展平图片
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x
```

### 4. 训练循环（带验证集）

```python
torch.manual_seed(1337)
model = MLP([784, 256, 256, 10])
criterion = nn.CrossEntropyLoss()                       # 多分类标配
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    # ===== 训练 =====
    model.train()                                       # 训练模式（影响 Dropout/BN）
    loss_train = 0
    for images, labels in train_loader:
        preds = model(images)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    # ===== 验证 =====
    model.eval()                                        # 验证模式
    loss_val = 0
    for images, labels in val_loader:
        with torch.no_grad():                           # 不建计算图
            loss = criterion(model(images), labels)
            loss_val += loss.item()

    print(f"Epoch {epoch}: train_loss={loss_train/len(train_loader):.4f}, "
          f"val_loss={loss_val/len(val_loader):.4f}")
```

### 5. 测试集评估

```python
correct, total = 0, 0
for images, labels in test_loader:
    with torch.no_grad():
        preds = model(images)
        _, predicted = torch.max(preds, 1)              # 取 10 个分数中最大值的索引
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试准确率: {correct / total * 100:.2f}%")
```

---

## 四、新概念

### 3.1 CrossEntropyLoss = Softmax + 负对数似然

```
模型输出 10 个 raw scores → [内部做 Softmax] → 转成概率 → 取正确类别的 -log(概率)
```

**不需要在模型最后一层加 Softmax**，`CrossEntropyLoss` 内部自带。

### 3.2 训练集 / 验证集 / 测试集

| 数据集 | 用途 | 训练时更新参数？ |
|---|---|---|
| 训练集 | 学习 | ✅ 是 |
| 验证集 | 监控过拟合、调超参 | ❌ 否，仅观察 |
| 测试集 | 最终成绩 | ❌ 否，只用一次 |

### 3.3 过拟合信号

```
train_loss 一直降，val_loss 不降反升 → 模型在背答案，没在学
```

### 3.4 L2 正则化 — 一行搞定

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
#                                                    ↑ 惩罚大权重
```

原理：`总 loss = CrossEntropyLoss + λ × Σ(权重²)`，逼模型权重保持小、少依赖个别特征。

### 3.5 Adam vs SGD

| | SGD | Adam |
|---|---|---|
| 学习率 | 固定，需要手动调 | 每个参数自适应 |
| 需要 scheduler | 一般需要 | 通常不需要 |
| 默认适用 | 论文复现 | **日常第一选择** |

---

## 五、`torch.max` 解析

```python
preds = [[0.12, 0.3, 0.8, ..., 1.5],     # 第 0 张图，10 个分数
         [1.2,  0.1, 0.5, ..., 0.3],     # 第 1 张图
         ...]                               # 共 64 张图

# torch.max(preds, 1)：沿每一行（第 1 维）找最大
# 返回 (最大值, 索引)
_, predicted = torch.max(preds, 1)

# predicted = [9, 0, 3, ...]  ← 每张图预测的数字
```

对比 Day 1：`score > 0` 判断正负 → 这里 `argmax` 判断 0~9。

---

## 六、两种过拟合方案对比（Dropout / BatchNorm）

### Dropout

```python
class MLP_Dropout(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        sz = layer_sizes
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(len(sz) - 1):
            self.layers.append(nn.Linear(sz[i], sz[i+1]))
            if i < len(sz) - 2:
                self.dropouts.append(nn.Dropout(0.2))   # 随机 20% 神经元失活

    def forward(self, x):
        x = x.view(-1, 28*28)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropouts[i](x)
                x = F.relu(x)
        return x
```

### BatchNorm

```python
class MLP_BN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        sz = layer_sizes
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(len(sz) - 1):
            self.layers.append(nn.Linear(sz[i], sz[i+1]))
            if i < len(sz) - 2:
                self.bns.append(nn.BatchNorm1d(sz[i+1]))  # 归一化到均值 0 方差 1

    def forward(self, x):
        x = x.view(-1, 28*28)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.bns[i](x)          # BN 在激活之前
                x = F.relu(x)
        return x
```

---

## 七、新踩的坑

| 坑 | 解决 |
|---|---|
| 图片四维 `(B,1,28,28)` 塞不进 Linear | `x.view(-1, 28*28)` 展平 |
| `_, predicted = torch.max(preds,1)` 不懂 | `_` 丢掉最大值，`predicted` 取索引 |
| `labels.size(0)` 是什么 | batch 样本数 |
| 跑多次结果变了 | `torch.manual_seed(…)` 固定种子；`model=MLP()` 放训练 cell |
| `model.train()` / `model.eval()` 漏了 | Dropout/BN 的开关，训练一定 `train()`，验证测试一定 `eval()` |

---

## 八、完整代码（最终版）

```python
# ===== 导入 =====
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader

# ===== 数据 =====
transform = T.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_set, val_set = torch.utils.data.random_split(train_data, [0.8, 0.2])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# ===== 模型 =====
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        sz = layer_sizes
        self.layers = nn.ModuleList([nn.Linear(sz[i], sz[i+1]) for i in range(len(sz)-1)])
    def forward(self, x):
        x = x.view(-1, 28*28)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1: x = F.relu(x)
        return x

# ===== 训练 =====
torch.manual_seed(1337)
model = MLP([784, 256, 256, 10])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

for epoch in range(5):
    model.train()
    loss_train = 0
    for images, labels in train_loader:
        loss = criterion(model(images), labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_train += loss.item()

    model.eval()
    loss_val = 0
    for images, labels in val_loader:
        with torch.no_grad():
            loss_val += criterion(model(images), labels).item()

    print(f"Epoch {epoch}: train_loss={loss_train/len(train_loader):.4f}, val_loss={loss_val/len(val_loader):.4f}")

# ===== 测试 =====
correct, total = 0, 0
for images, labels in test_loader:
    with torch.no_grad():
        _, predicted = torch.max(model(images), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"测试准确率: {correct / total * 100:.2f}%")
```

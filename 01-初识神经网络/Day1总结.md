# Day1 总结：从手撕神经网络到 PyTorch

---

## 一、任务与架构

**二分类**，用 `make_moons` 月牙形数据，区分 ±1 两类。

|      | 手撕版 (micrograd)      | PyTorch 版                         |
| ---- | ----------------------- | ---------------------------------- |
| 张量 | `Value`（自建）         | `torch.Tensor`                     |
| 模型 | `MLP(2, [16, 16, 1])`   | `nn.Sequential` 或循环构建         |
| 激活 | `.relu()`               | `F.relu()`                         |
| 损失 | `max(0, 1-yi*scorei)`   | `(1-y*scores).clamp(min=0).mean()` |
| 更新 | `p.data -= lr * p.grad` | `optimizer.step()`                 |
| 数据 | `get_batch()`           | `DataLoader`                       |

---

## 二、核心代码范式

### 数据

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 模型（动态构建）

```python
class MLP(nn.Module):
    def __init__(self, layer_sizes):                # 例: [784, 256, 256, 10]
        super().__init__()
        sz = layer_sizes
        self.layers = nn.ModuleList([
            nn.Linear(sz[i], sz[i+1])
            for i in range(len(sz) - 1)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:            # 最后一层不加激活
                x = F.relu(x)
        return x
```

### 训练循环（五步法）

```python
torch.manual_seed(1337)                    # 固定随机种子
model = MLP([...])

# 损失 & 优化器
loss_fn = nn.CrossEntropyLoss()            # 多分类
# loss = (1 - y * scores).clamp(min=0).mean()   # Hinge Loss（二分类 ±1）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        # ① 前向传播
        preds = model(batch_X)

        # ② 计算损失
        loss = loss_fn(preds, batch_y)

        # ③ 清零梯度
        optimizer.zero_grad()

        # ④ 反向传播
        loss.backward()

        # ⑤ 更新参数
        optimizer.step()
```

---

## 三、张量维度速查

| 操作       | 代码                      | 示例                        |
| ---------- | ------------------------- | --------------------------- |
| 看形状     | `tensor.shape`            | `torch.Size([32, 2])`       |
| 展平图片   | `x.view(-1, 28*28)`       | `(64,1,28,28)` → `(64,784)` |
| 插入维度   | `tensor.unsqueeze(1)`     | `(32,)` → `(32, 1)`         |
| 去除维度   | `tensor.squeeze()`        | `(32, 1)` → `(32,)`         |
| 取数值     | `tensor.item()`           | 标量张量 → Python 数字      |
| 断开计算图 | `tensor.detach()`         | 不参与梯度                  |
| 转 NumPy   | `tensor.detach().numpy()` | 安全写法                    |

---

## 四、踩过的坑

| 坑                                  | 解决                             |
| ----------------------------------- | -------------------------------- |
| `numel` 当成属性                    | 是方法：`numel()`                |
| MSE 做分类                          | Hinge Loss 或 `CrossEntropyLoss` |
| `loss_epoch += loss` 漏了 `.item()` | `loss.item()`                    |
| scheduler 放 batch 里               | 放 epoch 层面                    |
| `.data` 废弃                        | 用 `.detach()`                   |
| 重复跑 cell 不重初始化              | `model=MLP()` 和训练放同一 cell  |
| 不设随机种子                        | `torch.manual_seed(1337)`        |

---

## 五、完整代码

```python
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F

# 数据
X, y = make_moons(n_samples=100, noise=0.1)
X, y = torch.tensor(X).float(), torch.tensor(y).float()
y = y * 2 - 1
dataloader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# 模型
class MLP(nn.Module):
    def __init__(self, layer_sizes): super().__init__()
        sz = layer_sizes
        self.layers = nn.ModuleList([nn.Linear(sz[i], sz[i+1]) for i in range(len(sz)-1)])
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers)-1 else layer(x)
        return x

# 训练
torch.manual_seed(1337)
model = MLP([2, 16, 16, 1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    loss_epoch, correct, total = 0, 0, 0
    for Xb, yb in dataloader:
        scores = model(Xb)
        loss = (1 - yb.unsqueeze(1) * scores).clamp(min=0).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_epoch += loss.item()
        correct += ((scores > 0) == (yb.unsqueeze(1) > 0)).sum().item()
        total += len(yb)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss_epoch:.3f}, acc={correct/total*100:.1f}%")
```

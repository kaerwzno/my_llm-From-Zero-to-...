import torch
import torch.nn as nn
from torch.nn import functional as F

# ── 超参数 ──────────────────────────────────────────────────────────────────
block_size   = 256    # 每次能看多长的上下文（序列长度 T）
batch_size   = 64  # 一次训练多少条序列
max_iters    = 5000
eval_interval= 500
learning_rate= 3e-4
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters   = 200
n_embd       = 384   # 每个 token 的向量维度（C）
n_head       = 6     # 注意力头数；每个头的维度 = n_embd // n_head = 64
n_layer      = 6    # 堆几个 Transformer Block
dropout      = 0.3  # 每次训练随机丢弃 20% 的神经元，缓解过拟合
# ────────────────────────────────────────────────────────────────────────────

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars      = sorted(list(set(text)))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda l: ''.join(itos[i] for i in l)

data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i + block_size]     for i in ix])
    y    = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    # @torch.no_grad() 比 with torch.no_grad() 放在函数外面更干净
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y     = get_batch(split)
            _, loss  = model(X, Y)
            losses[k]= loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. 单个自注意力头
#    - 负责计算"谁应该注意谁"，以及把 value 加权求和传出去
#    - 这就是之前 notebook 里 wei = tril(...) @ x 的可学习升级版
# ══════════════════════════════════════════════════════════════════════════════
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # 三个线性层：同一个 x 投影出三个不同身份
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # 我有什么（广告牌）
        self.query = nn.Linear(n_embd, head_size, bias=False)  # 我在找什么
        self.value = nn.Linear(n_embd, head_size, bias=False)  # 我实际能给什么

        # tril 是不可训练的常量，注册为 buffer（会随模型保存，但不求梯度）
        # 形状 (block_size, block_size)，用来实现 causal masking：
        # 每个位置只能看过去，不能看未来
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # 注意力权重上的 dropout：随机丢掉部分 token 间的连接，防止过拟合
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape           # C 此处是 head_size

        k = self.key(x)             # (B, T, head_size)
        q = self.query(x)           # (B, T, head_size)

        # 相似度矩阵：q 和 k 的点积；结果 [b, t, i] = q_t · k_i
        # 代表"位置 t 对位置 i 的兴趣程度"
        wei = q @ k.transpose(-2, -1)            # (B, T, T)

        # 缩放：点积数值随维度增大而增大，除以 √C 防止 softmax 饱和（梯度消失）
        wei = wei * C**-0.5

        # Causal masking：把上三角（未来位置）填成 -inf，softmax 后变 0
        # [:T, :T] 切片让它在 generate 时 T < block_size 也能正常工作
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax 让每一行的权重和为 1，变成"注意力分布"
        wei = F.softmax(wei, dim=-1)             # (B, T, T)
        wei = self.attn_dropout(wei)             # 随机丢掉部分注意力连接

        # value 才是真正被传输的内容；key 到这里就不出现了
        v   = self.value(x)                      # (B, T, head_size)
        out = wei @ v                            # (B, T, head_size)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. 多头注意力
#    - 多个 Head 并行跑，每个头学不同的"关系"（比如一个学语法、一个学语义）
#    - 结果沿 C 维度拼接，再用一个线性层投影回 n_embd
# ══════════════════════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # proj：把多头拼接后的向量投影回 n_embd，维度不变，让残差连接可以直接相加
        self.proj  = nn.Linear(n_embd, n_embd)
        # 残差路径上的 dropout：在 proj 输出加回 x 之前随机丢弃
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 每个 head 输出 (B, T, head_size)，cat 后变 (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.resid_dropout(out)   # 残差连接前随机丢弃，缓解过拟合
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. 前馈网络（Feed Forward）
#    - 注意力负责"token 之间通信"，前馈负责每个 token 自己消化信息
#    - 先放大 4 倍再收回，给模型足够的非线性容量
# ══════════════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 放大
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 收回
            nn.Dropout(dropout),            # 前馈网络输出随机丢弃
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Transformer Block
#    - 把注意力 + 前馈组合成一块，加上残差连接和 LayerNorm
#    - 残差连接：x = x + f(x)
#      好处：梯度有"高速公路"直通早期层，深网络才能训得动
#    - LayerNorm：在每个 token 的 C 个维度上归一化（不同于 BatchNorm 的跨样本归一化）
#      这里用的是 pre-norm（先 LN 再注意力），比原始论文的 post-norm 更稳定
# ══════════════════════════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size  = n_embd // n_head       # 每个头的维度，例如 32 // 4 = 8
        self.sa    = MultiHeadAttention(n_head, head_size)
        self.ffwd  = FeedForward(n_embd)
        self.ln1   = nn.LayerNorm(n_embd)   # 注意力前的归一化
        self.ln2   = nn.LayerNorm(n_embd)   # 前馈前的归一化

    def forward(self, x):
        x = x + self.sa(self.ln1(x))        # 通信：token 之间交换信息
        x = x + self.ffwd(self.ln2(x))      # 计算：每个 token 自己消化
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 5. 完整模型（从 Bigram 升级为简版 GPT）
#    - 和之前相比：x 在进 lm_head 之前，先经过 n_layer 个 Block
#    - generate 里加了 idx_cond 截断，防止序列超过 block_size 时 pos_emb 越界
# ══════════════════════════════════════════════════════════════════════════════
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 堆叠 n_layer 个 Block，让信息逐层抽象
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        # 最后一个 LayerNorm：所有 Block 跑完后再统一归一化一次
        self.ln_f   = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                               # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x       = tok_emb + pos_emb    # token 内容 + 位置，(B, T, n_embd)
        x       = self.blocks(x)       # 经过所有 Block，token 之间互相通信
        x       = self.ln_f(x)         # 最终归一化
        logits  = self.lm_head(x)      # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 截断：只取最后 block_size 个 token，防止 position_embedding 越界
            idx_cond       = idx[:, -block_size:]
            logits, _      = self(idx_cond, targets=None)
            logits         = logits[:, -1, :]              # 只取最后一个位置的预测
            probs          = F.softmax(logits, dim=-1)
            idx_next       = torch.multinomial(probs, num_samples=1)
            idx            = torch.cat((idx, idx_next), dim=1)
        return idx


# ── 训练 ─────────────────────────────────────────────────────────────────────
model = Bigram(vocab_size)
m     = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb     = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(m.state_dict(), 'model.pt')
print("模型已保存到 model.pt")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated)
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(generated)

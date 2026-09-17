import torch
import torch.nn as nn
from torch.nn import functional as F

# ── 超参数：必须和训练时一致，否则加载会报错 ──────────────────────────────
block_size   = 256
n_embd       = 384
n_head       = 6
n_layer      = 6
dropout      = 0.2
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
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


# ═══ 模型结构：必须和 bigram.py 里完全一致 ═══
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        v   = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.resid_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa    = MultiHeadAttention(n_head, head_size)
        self.ffwd  = FeedForward(n_embd)
        self.ln1   = nn.LayerNorm(n_embd)
        self.ln2   = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        # temperature < 1.0 更保守（只选高概率词），> 1.0 更随机
        for _ in range(max_new_tokens):
            idx_cond       = idx[:, -block_size:]
            logits, _      = self(idx_cond, targets=None)
            logits         = logits[:, -1, :] / temperature
            probs          = F.softmax(logits, dim=-1)
            idx_next       = torch.multinomial(probs, num_samples=1)
            idx            = torch.cat((idx, idx_next), dim=1)
        return idx


# ── 加载已训练模型 ─────────────────────────────────────────────────────────
model = Bigram(vocab_size)
model.load_state_dict(torch.load('model.pt', map_location=device))
m = model.to(device)
m.eval()   # 关键：切换到 eval 模式，关闭 dropout


# ── 生成 ────────────────────────────────────────────────────────────────────
# 起始上下文：给的越多，生成越能"顺着"开头往下写
context = encode("\n")   # 用换行开头，模拟空白起始
context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

# max_new_tokens 想生成多少有多少
generated = decode(m.generate(context, max_new_tokens=2000, temperature=0.8)[0].tolist())
print(generated)

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(generated)

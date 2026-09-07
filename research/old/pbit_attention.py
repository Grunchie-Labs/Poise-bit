# ============================================================
# Tiny Transformer Baseline
# Designed for experimentation with:
# - stochastic attention
# - p-bit routing
# - thermodynamic temperature
# - entropy analysis
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    vocab_size = 100
    seq_len = 32

    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 256

    batch_size = 32
    lr = 3e-4
    epochs = 10

    dropout = 0.1

    # IMPORTANT FOR YOU
    temperature = 1.0


cfg = Config()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SYNTHETIC DATA
# simple next-token prediction
# ============================================================

def generate_data(batch_size, seq_len, vocab_size):
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.roll(x, shifts=-1, dims=1)
    return x.to(device), y.to(device)

# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ============================================================
# MULTIHEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads

        self.temperature = cfg.temperature

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ====================================================
        # ATTENTION SCORES
        # THIS IS YOUR PLAYGROUND
        # ====================================================
        p_bit_weight=
        scores = ((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim))

        # ====================================================
        # THERMODYNAMIC TEMPERATURE
        # ====================================================

        scores = scores / self.temperature

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # ====================================================
        # ATTENTION PROBABILITIES
        # ====================================================

        attn = F.softmax(scores, dim=-1)

        # ====================================================
        # METRICS
        # ====================================================

        entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean()

        sparsity = (attn < 0.01).float().mean()

        max_attention = attn.max().item()

        attn = self.dropout(attn)

        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(out)

        metrics = {
            "entropy": entropy.item(),
            "sparsity": sparsity.item(),
            "max_attention": max_attention
        }

        return out, metrics

# ============================================================
# FEEDFORWARD
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = MultiHeadAttention(cfg)

        self.ff = FeedForward(cfg)

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x):

        attn_out, metrics = self.attn(self.ln1(x))

        x = x + attn_out

        x = x + self.ff(self.ln2(x))

        return x, metrics

# ============================================================
# TRANSFORMER MODEL
# ============================================================

class TinyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.positional = PositionalEncoding(cfg.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg)
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        x = self.positional(x)

        all_metrics = []

        for block in self.blocks:
            x, metrics = block(x)
            all_metrics.append(metrics)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits, all_metrics

# ============================================================
# TRAINING
# ============================================================

model = TinyTransformer(cfg).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

print("\nTraining Started\n")

for epoch in range(cfg.epochs):

    x, y = generate_data(
        cfg.batch_size,
        cfg.seq_len,
        cfg.vocab_size
    )

    logits, metrics = model(x)

    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # ========================================================
    # PRINT METRICS
    # ========================================================

    print(f"\nEpoch {epoch+1}")

    print(f"Loss: {loss.item():.4f}")

    for i, m in enumerate(metrics):

        print(f"\nBlock {i}")

        print(f"Entropy:      {m['entropy']:.4f}")
        print(f"Sparsity:     {m['sparsity']:.4f}")
        print(f"Max Attention:{m['max_attention']:.4f}")

# ============================================================
# THINGS YOU SHOULD EXPERIMENT WITH
# ============================================================

"""
1. STOCHASTIC SCORES

scores = scores + noise

noise = torch.randn_like(scores) * sigma

------------------------------------------------

2. P-BIT EDGE SAMPLING

prob = torch.sigmoid(scores / T)

mask = torch.bernoulli(prob)

scores = scores * mask

------------------------------------------------

3. DYNAMIC TEMPERATURE

T = f(entropy)

T = learnable

T = per-head

T = per-token

------------------------------------------------

4. ISING-LIKE COUPLINGS

scores = J * scores

where J evolves dynamically

------------------------------------------------

5. ANNEALING

temperature decay during training

------------------------------------------------

6. ATTENTION ENTROPY TRACKING

Very important for your research

------------------------------------------------
"""
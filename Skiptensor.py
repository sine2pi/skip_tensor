import torch
import torch.nn as nn
import torch.nn.functional as F

class Skiptensor(nn.Module):
    def __init__(self, dims, head, skip_pattern=None, learned_skip=False, max_skip=8):
        super().__init__()
        self.dims = dims
        self.head = head
        self.skip_pattern = skip_pattern
        self.learned_skip = learned_skip
        self.max_skip = max_skip
        self.scale = dims ** -0.5
        
        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims)
        self.value = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)
        
        if learned_skip:
            self.skip_gate = nn.Sequential(nn.Linear(dims, dims), nn.ReLU(), nn.Linear(dims, max_skip + 1))

    def forward(self, x, layer_idx=None, skip_pattern=None):
        batch_size, seq_len, _ = x.shape
        q = self.query(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)

        if self.learned_skip:
            pooled_input = x.mean(dim=1)
            skip_logits = self.skip_gate(pooled_input)
            skip_weights = F.gumbel_softmax(skip_logits, hard=True)
            skip_pattern = torch.argmax(skip_weights, dim=-1) + 1
            # For simplicity, use the first in batch (or broadcast) for all heads
            skip_pattern = skip_pattern[0].item() if isinstance(skip_pattern, torch.Tensor) else int(skip_pattern)
            
        elif skip_pattern is not None:
            skip_pattern = skip_pattern
        elif self.skip_pattern is not None:
            skip_pattern = self.skip_pattern
        elif layer_idx is not None:
            skip_pattern = max(1, 6 - layer_idx)
        else:
            skip_pattern = 1

        k_skipped = k[:, :, ::skip_pattern, :]
        v_skipped = v[:, :, ::skip_pattern, :]

        attn_scores = torch.matmul(q, k_skipped.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v_skipped)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dims)
        return self.output(output)

class Transformer(nn.Module):
    def __init__(self, dims, head, skip_pattern=None, learned_skip=False):
        super().__init__()
        
        self.attention = Skiptensor(dims, head, skip_pattern=skip_pattern, learned_skip=learned_skip)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.ffn = nn.Sequential(nn.Linear(dims, dims), nn.ReLU(), nn.Linear(dims, dims))

    def forward(self, x, layer_idx=None, skip_pattern=None):
        x = x + self.attention(self.norm1(x), layer_idx=layer_idx, skip_pattern=skip_pattern)
        x = x + self.ffn(self.norm2(x))
        return x

class AudioModel(nn.Module):
    def __init__(self, dims, head, skip_patterns=None, learned_skip_layers=None, num_layers=4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            learned = learned_skip_layers and i in learned_skip_layers
            pattern = skip_patterns[i] if skip_patterns else None
            self.layers.append(Transformer(dims, head, skip_pattern=pattern, learned_skip=learned))

    def forward(self, x, skip_patterns_override=None):
        
        for i, layer in enumerate(self.layers):
            pattern = skip_patterns_override[i] if skip_patterns_override else None
            x = layer(x, layer_idx=i, skip_pattern=pattern)
            
        return x

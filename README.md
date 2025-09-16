``` python

import torch
import torch.nn as nn
import math

class Skiptensor(nn.Module):
    def __init__(self, dims, head, skip_pattern=1):
        super().__init__()
        self.dims = dims
        self.head = head
        self.skip_pattern = skip_pattern
        self.scale = dims ** -0.5
        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims)
        self.value = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)

    def forward(self, x, layer_index, skip_pattern=None):

        batch_size, seq_len, _ = x.shape
        q = self.query(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)
        
        if skip_pattern is None:
            self.skip_pattern = max(1, 4 - layer_index) # some pattern
        else:
            self.skip_pattern = skip_pattern

        k_skipped = k[:, :, ::self.skip_pattern, :]
        v_skipped = v[:, :, ::self.skip_pattern, :]

        attn_scores = torch.matmul(q, k_skipped.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v_skipped)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dims)

        return self.output(output)

class Transformer(nn.Module):
    def __init__(self, dims, head):
        super().__init__()

        self.attention = Skiptensor(dims, head, skip_pattern=1)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.ffn = nn.Sequential(nn.Linear(dims, dims), nn.ReLU(), nn.Linear(dims, dims))

    def forward(self, x, layer_index):
        x = x + self.attention(self.norm1(x), layer_index)
        x = x + self.ffn(self.norm2(x))
        return x

class AudioModel(nn.Module):
    def __init__(self, dims, head, layer=4):
        super().__init__()

        self.layers = nn.ModuleList([
            Transformer(dims, head, Skiptensor(dims, head, skip_pattern=4)),  # Layer 1: Attends to every 4th tensor
            Transformer(dims, head, Skiptensor(dims, head, skip_pattern=3)),  # Layer 2: Attends to every 3rd tensor
            Transformer(dims, head, Skiptensor(dims, head, skip_pattern=2)),  # Layer 3: Attends to every other tensor
            Transformer(dims, head, Skiptensor(dims, head, skip_pattern=1))   # Layer 4: Attends to every tensor
        for _ in range(layer)])  

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            x = layer(x, layer_index=i)

        return x

####


class LearnedSkip(nn.Module):
    def __init__(self, dims, head, max_skip=8):
        super().__init__()
        self.dims = dims
        self.head = head
        self.scale = dims ** -0.5
        self.max_skip = max_skip
        
        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims)
        self.value = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)
        
        self.skip_gate = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, max_skip + 1)  # Output a score for each possible skip pattern (1 to max_skip)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        pooled_input = x.mean(dim=1)  # Shape: (batch_size, dims)
        skip_logits = self.skip_gate(pooled_input)

        skip_weights = F.gumbel_softmax(skip_logits, hard=True)
        skip_pattern = torch.argmax(skip_weights, dim=-1) + 1  # Add 1 as skip patterns start from 1
        
        q = self.query(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.head, self.dims // self.head).transpose(1, 2)

        mask = self._create_skip_mask(seq_len, skip_pattern, device=x.device)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores.masked_fill_(mask == 0, float('-inf'))  # Mask out the skipped indices
        
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dims)
        
        return self.output(output)

    def _create_skip_mask(self, seq_len, skip_patterns, device):
        mask = torch.zeros(len(skip_patterns), seq_len, seq_len, device=device, dtype=torch.bool)
        for i, skip in enumerate(skip_patterns):
            for j in range(0, seq_len, skip.item()):
                mask[i, :, j] = True
        return mask.unsqueeze(1) # Add a head dimension



```        

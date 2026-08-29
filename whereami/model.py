from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ContextGRU(nn.Module):
    def __init__(self, hidden_size: int = 24):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, symbols: torch.Tensor, h0: torch.Tensor | None = None):
        x = F.one_hot(symbols.long(), num_classes=3).float()
        if h0 is not None and h0.ndim == 2:
            h0 = h0.unsqueeze(0)
        states, hn = self.gru(x, h0)
        logits = self.head(states)
        return logits, states, hn

    def step(self, symbol: int, h: torch.Tensor):
        """One recurrent step from an explicit hidden state (for interventions/Jacobians)."""
        if h.ndim == 1:
            h = h.unsqueeze(0)
        sym = torch.full((h.shape[0], 1), int(symbol), device=h.device, dtype=torch.long)
        logits, states, _ = self.forward(sym, h)
        return logits[:, 0], states[:, 0]

import torch
import torch.nn as nn


class PoolingLogits(nn.Module):
    def __init__(self, hidden_size, layer_norm=True):
        super(PoolingLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        if self.layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states



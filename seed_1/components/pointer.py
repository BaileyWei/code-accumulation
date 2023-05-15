from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class PointerNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_size = config.kernel_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.We = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wd = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.Wo = nn.Linear(hidden_size, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: [batch_size, seq_len, hidden_size]
        decoder_hidden: [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = encoder_outputs.size()

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_items = self.dropout(self.Wd(decoder_hidden))

        encoder_items = self.dropout(self.We(encoder_outputs))

        # cross_items = self.dropout(self.Wo(encoder_outputs) * decoder_items)
        # energy = self.v(torch.tanh(decoder_items + encoder_items + cross_items + self.b)).squeeze()
        energy = self.v(torch.tanh(decoder_items + encoder_items + self.b)).squeeze()

        return energy


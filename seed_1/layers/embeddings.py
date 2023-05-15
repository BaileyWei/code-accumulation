from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class RotaryPositionalEmbedding(nn.Embedding):
    """
    Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
    This module produces sinusoidal positional embeddings of any length.
    """
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)

        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """

        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()

        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seq_len]."""

        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )

        return super().forward(positions)


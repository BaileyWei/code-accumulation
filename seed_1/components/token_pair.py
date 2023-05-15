import torch
import torch.nn as nn

from seed.layers.normalization import ConditionalLayerNormalization
from seed.layers.embeddings import RotaryPositionalEmbedding


class TokenPairComponent(nn.Module):
    def __init__(self, config):
        super(TokenPairComponent, self).__init__()

        self.config = config
        self.config.update(config.task_specific_params)
        token_pair_kernel = config.token_pair_kernel
        assert token_pair_kernel in {"TP", "TPP", "GP", "EGP"}
        if token_pair_kernel == "TP":
            self.token_pair_kernel = TPLinkerKernel(config)
        elif token_pair_kernel == "TPP":
            self.token_pair_kernel = TPLinkerPlusKernel(config)
        elif token_pair_kernel == "GP":
            self.token_pair_kernel = GlobalPointerKernel(config)
        elif token_pair_kernel == "EGP":
            self.token_pair_kernel = EfficientGlobalPointerKernel(config)

    def forward(self, input: torch.FloatTensor):

        seq_len = input.size()[-2]
        start_token_list, end_token_list = list(), list()
        for index in range(seq_len):
            start_token_hidden = input[:, index, :]
            end_token_hidden = input[:, index:, :]  # index: only look back
            start_token_hidden = start_token_hidden[:, None, :].repeat(1, seq_len - index, 1)

            start_token_list.append(start_token_hidden)
            end_token_list.append(end_token_hidden)

        start_token_hidden = torch.cat(start_token_list, dim=-2)
        end_token_hidden = torch.cat(end_token_list, dim=-2)

        token_pair_hidden = self.token_pair_kernel(start_token_hidden, end_token_hidden)

        return token_pair_hidden


class TPLinkerKernel(nn.Module):
    def __init__(self, config):
        super(TPLinkerKernel, self).__init__()

        self.config = config
        self.token_pair_fusion = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.dropout_prob),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.output_size)
        )

    def forward(self, start_token_hidden, end_token_hidden):

        input_token_hidden = torch.cat([start_token_hidden, end_token_hidden], dim=-1)
        token_pair_hidden = self.token_pair_fusion(input_token_hidden)

        return token_pair_hidden


class TPLinkerPlusKernel(nn.Module):
    def __init__(self, config):
        super(TPLinkerPlusKernel, self).__init__()

        self.config = config
        self.cln = ConditionalLayerNormalization(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, start_token_hidden, end_token_hidden):

        token_pair_hidden = self.cln(end_token_hidden, start_token_hidden)
        token_pair_hidden = self.fc(self.dropout(token_pair_hidden))

        return token_pair_hidden


def apply_rotary(rope, x):

    sin, cos = rope(x.shape[2])[None, None, :, :].chunk(2, dim=-1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
    # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
    # torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GlobalPointerKernel(nn.Module):
    def __init__(self, config):
        super(GlobalPointerKernel, self).__init__()

        self.config = config
        self.query = nn.Linear(config.hidden_size, config.head_size * config.output_size)
        self.key = nn.Linear(config.hidden_size, config.head_size * config.output_size)
        self.rope = RotaryPositionalEmbedding(512, config.head_size) if config.rope else None
        self.apply_rotary = apply_rotary

    def forward(self, start_token_hidden, end_token_hidden):

        assert start_token_hidden.size() == end_token_hidden.size()
        batch_size, seq_len = start_token_hidden.size()[:2]

        query, key = self.query(start_token_hidden), self.key(end_token_hidden)
        query = query.view(batch_size, seq_len, self.config.output_size, self.config.head_size)
        key = key.view(batch_size, seq_len, self.config.output_size, self.config.head_size)

        if self.rope is not None:
            query = self.apply_rotary(self.rope, query)
            key = self.apply_rotary(self.rope, key)

        token_pair_hidden = torch.einsum("blnh,blnh->bln", query, key)

        return token_pair_hidden


class EfficientGlobalPointerKernel(nn.Module):
    def __init__(self, config):
        super(EfficientGlobalPointerKernel, self).__init__()

        self.config = config
        self.query = nn.Linear(config.hidden_size, config.head_size)
        self.key = nn.Linear(config.hidden_size, config.head_size)
        # assert self.config.entity_type_feature in {"original", "head"}
        # if self.config.entity_type_feature == "original":
        #     self.entity_type_fc = nn.Linear(config.hidden_size * 2, config.output_size)
        # elif self.config.entity_type_feature == "head":
        #     self.entity_type_fc = nn.Linear(config.head_size * 2, config.output_size)
        self.entity_type_bias = nn.Linear(config.head_size * 4, config.output_size)
        self.rope = RotaryPositionalEmbedding(512, config.head_size) if config.rope else None
        self.apply_rotary = apply_rotary

    def forward(self, start_token_hidden, end_token_hidden):

        assert start_token_hidden.size() == end_token_hidden.size()
        batch_size, seq_len = start_token_hidden.size()[:2]

        query_start, query_end = self.query(start_token_hidden), self.query(end_token_hidden)
        key_start, key_end = self.key(end_token_hidden), self.key(end_token_hidden)
        entity_type_feat = torch.cat([query_start, key_start, query_end, key_end], dim=-1)
        entity_type_feat = self.entity_type_bias(entity_type_feat)

        query = query_start.view(batch_size, seq_len, 1, self.config.head_size)
        key = key_end.view(batch_size, seq_len, 1, self.config.head_size)

        if self.rope is not None:
            query = self.apply_rotary(self.rope, query)
            key = self.apply_rotary(self.rope, key)

        token_pair_hidden = torch.einsum("blnh,blnh->bln", query, key)
        token_pair_hidden = token_pair_hidden + entity_type_feat

        return token_pair_hidden


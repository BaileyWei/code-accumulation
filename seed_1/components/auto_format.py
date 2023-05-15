import copy
from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.activations import ACT2FN

from seed.components.pointer import PointerNetwork
from seed.components.token_pair import TPLinkerPlusKernel


class AutoFormatComponent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.kernel_fc = nn.Linear(config.hidden_size, config.kernel_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.auto_format_kernel = nn.ModuleList(
            [AutoFormatLayer(config) for _ in range(config.num_layers)]
        )
        self.node_fc = nn.Linear(config.kernel_size, config.node_output_size)
        self.link_fc = nn.Linear(config.kernel_size * 2, config.link_output_size)

    def forward(self, input: torch.FloatTensor):

        seq_len = input.size()[-2]

        input = self.dropout(self.kernel_fc(input))
        for layer in self.auto_format_kernel:
            input = layer(input)

        node_output = self.node_fc(input)

        start_token_list, end_token_list = list(), list()
        for index in range(seq_len - 1):
            start_token_hidden = input[:, index, :]
            end_token_hidden = input[:, index + 1:, :]  # index: only look back and not include itself
            start_token_hidden = start_token_hidden[:, None, :].repeat(1, seq_len - index - 1, 1)

            start_token_list.append(start_token_hidden)
            end_token_list.append(end_token_hidden)

        start_token_hidden = torch.cat(start_token_list, dim=-2)
        end_token_hidden = torch.cat(end_token_list, dim=-2)

        link_output = self.link_fc(torch.concat([start_token_hidden, end_token_hidden], dim=-1))

        return node_output, link_output


class AutoFormatV2Component(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.kernel_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.auto_format_kernel = nn.ModuleList([AutoFormatLayer(config) for _ in range(config.num_layers)])

        self.au_classifier = nn.Linear(config.kernel_size, config.node_output_size)
        self.node_classifier = nn.Linear(config.kernel_size, config.num_labels)
        self.link_classifier = PointerNetwork(config)

    def forward(self, input: torch.FloatTensor):

        seq_len = input.size()[-2]

        input = self.input_fc(input)
        for layer in self.auto_format_kernel:
            input = layer(input)

        au_output = self.au_classifier(input)
        node_output = self.node_classifier(input)

        link_output = list()
        for index in range(seq_len):
            decoder_hidden = input[:, index, :]
            encoder_outputs = input
            link_output.append(self.link_classifier(encoder_outputs, decoder_hidden))

        link_output = torch.stack(link_output, dim=0)

        # link_output = link_output + \
        #               torch.triu(fill_with_neg_inf(torch.zeros([seq_len, seq_len])), 1).to(link_output.device)

        return au_output, node_output, link_output


class AutoFormatV3Component(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_fc = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.kernel_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.auto_format_kernel = nn.ModuleList([AutoFormatLayer(config) for _ in range(config.num_layers)])

        self.au_classifier = nn.Linear(config.kernel_size, config.node_output_size)
        self.node_classifier = nn.Linear(config.kernel_size, config.num_labels)

        tp_config = copy.deepcopy(config)
        tp_config.hidden_size = config.kernel_size
        tp_config.dropout_prob = config.hidden_dropout_prob
        tp_config.output_size = config.link_output_size
        self.link_classifier = TPLinkerPlusKernel(tp_config)

    def forward(self, input: torch.FloatTensor):

        seq_len = input.size()[-2]

        input = self.input_fc(input)
        for layer in self.auto_format_kernel:
            input = layer(input)

        au_output = self.au_classifier(input)
        node_output = self.node_classifier(input)

        start_token_list, end_token_list = list(), list()
        for index in range(seq_len - 1):
            start_token_hidden = input[:, index, :]
            end_token_hidden = input[:, index + 1:, :]  # index: only look back and not include itself
            start_token_hidden = start_token_hidden[:, None, :].repeat(1, seq_len - index - 1, 1)

            start_token_list.append(start_token_hidden)
            end_token_list.append(end_token_hidden)

        start_token_hidden = torch.cat(start_token_list, dim=-2)
        end_token_hidden = torch.cat(end_token_list, dim=-2)

        link_output = self.link_classifier(start_token_hidden, end_token_hidden)

        return au_output, node_output, link_output


class AutoFormatLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor
    ) -> torch.FloatTensor:

        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        return layer_output

    def feed_forward_chunk(self, attention_output):

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.attention_type == 'normal':
            self.self = AttentionKernel(config)
        elif config.attention_type == 'non_symmetric':
            self.self = NonSymmetricAttentionKernel(config)
        self.output = AttentionOutput(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:

        self_outputs = self.self(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class AttentionKernel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.qkv = nn.Linear(
            config.kernel_size, config.num_heads * (2 * config.attention_size + config.value_size), bias=False
        )
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.split_list = [config.num_heads * config.attention_size,
                           config.num_heads * config.attention_size,
                           config.num_heads * config.value_size]
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            node: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        update1: 实现了Attention mask, 但没有去掉batch size的限制, 因为配套还没有跟上
        :param node: batch_size x seq_len x embed_dim
        # :param mask: seq_len x seq_len
        :return:
        """
        batch_size, seq_len, embed_dim = node.size()

        assert embed_dim == self.config.kernel_size

        q, k, v = torch.split(self.qkv(node), self.split_list, dim=-1)
        # batch_size x n_head x seq_len x head_dim
        q = q.view(batch_size, seq_len, self.config.num_heads, self.config.attention_size).transpose(1, 2)
        # batch_size x n_head x head_dim x seq_len
        k = k.view(batch_size, seq_len, self.config.num_heads, self.config.attention_size).permute(0, 2, 3, 1)
        # batch_size x n_head x seq_len x head_dim
        v = v.view(batch_size, seq_len, self.config.num_heads, self.config.attention_size).transpose(1, 2)

        att = torch.matmul(q, k)  # batch_size x n_head x seq_len x seq_len
        att /= (self.config.attention_size ** 0.5)
        att = self.attention_dropout(att)
        # 如果有mask应给是加在这里
        # if mask is not None:
        #     assert mask.shape[-2:] == (seq_len, seq_len)
        #     att.masked_fill_(mask=mask, value=float('-inf'))  # mask可以自己广播
        att = torch.softmax(att, dim=-1)
        att = self.hidden_dropout(att)

        v = torch.matmul(att, v)  # batch_size x n_head x seq_len x value_dim
        v = v.transpose(1, 2).reshape(batch_size, seq_len, -1)
        v = self.w(v)

        return v


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n)  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(
            2 * closest_power_of_2)[0::2][:n - closest_power_of_2]


class NonSymmetricAttentionKernel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_heads = config.num_heads
        self.attention_size = config.attention_size
        self.all_head_size = self.num_heads * self.attention_size

        self.query = nn.Linear(config.kernel_size, self.all_head_size)
        self.key = nn.Linear(config.kernel_size, self.all_head_size)
        self.value = nn.Linear(config.kernel_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:

        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_size)
        x = x.view(new_x_shape)

        return x.permute(0, 2, 1, 3)

    def get_alibi_bias(self, max_position: int) -> torch.Tensor:

        num_heads = self.num_heads
        _future_mask_right = torch.triu(fill_with_neg_inf(torch.zeros([max_position, max_position])), 1).\
            unsqueeze(0).repeat(num_heads // 3, 1, 1)
        _future_mask_left = torch.tril(fill_with_neg_inf(torch.zeros([max_position, max_position])), -1).\
            unsqueeze(0).repeat(num_heads // 3, 1, 1)
        _future_mask = torch.zeros([max_position, max_position]).unsqueeze(0).repeat(num_heads // 3, 1, 1)

        non_sym_mask = torch.cat([_future_mask_right, _future_mask_left, _future_mask], dim=0).unsqueeze(0)
        slopes = torch.Tensor(get_slopes(num_heads // 3)) * -1

        context_position = torch.arange(max_position)[:, None] ** 0.5
        memory_position = torch.arange(max_position)[None, :] ** 0.5

        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(num_heads // 3, -1, -1)

        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.view(1, num_heads // 3, max_position, max_position)
        alibi = alibi.repeat(1, 3, 1, 1)

        return alibi + non_sym_mask

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> torch.FloatTensor:

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_size)
        attention_scores = attention_scores + self.get_alibi_bias(attention_scores.size()[-1]).to(attention_scores.device)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class AttentionOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_heads
        self.attention_size = config.attention_size
        self.all_head_size = self.num_heads * self.attention_size
        self.dense = nn.Linear(self.all_head_size, config.kernel_size)
        self.LayerNorm = nn.LayerNorm(config.kernel_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.kernel_size, config.kernel_size * 4)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.kernel_size * 4, config.kernel_size)
        self.LayerNorm = nn.LayerNorm(config.kernel_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


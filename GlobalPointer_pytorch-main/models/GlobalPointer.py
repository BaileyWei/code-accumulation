import sys
# sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn


# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#         self.length = len(data)
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return self.length
#
#
# class DataMaker:
#     def __init__(self, tokenizer, add_special_tokens=True):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.add_special_tokens = add_special_tokens
#         self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)
#
#
#     def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
#         """生成喂入模型的数据
#
#         Args:
#             datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
#             max_seq_len (int): 句子最大token数量
#             ent2id (dict): ent到id的映射
#             data_type (str, optional): data类型. Defaults to "train".
#
#         Returns:
#             list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
#         """
#
#         ent_type_size = len(ent2id)  # 实体类别
#
#         all_inputs = []
#         for sample in datas:
#
#             inputs = self.tokenizer(
#                 sample["text"],
#                 max_length=max_seq_len,
#                 truncation=True,
#                 padding='max_length',
#                 return_offsets_mapping=True
#             )
#             if 'chinese' not in self.tokenizer.name_or_path:
#                 language = 'english'
#             else:
#                 language = 'chinese'
#
#             labels = None
#             if data_type != "test":
#                 ent2token_spans = self.preprocessor.get_ent2token_spans(
#                     sample["text"], sample["entity_list"], language=language
#                 )
#                 labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
#                 for start, end, label in ent2token_spans:
#                     if start >= 512 or end >= 512:
#                         continue
#                         # print(start, end)
#                     labels[ent2id[label], start, end] = 1
#             inputs["labels"] = labels
#
#             input_ids = torch.tensor(inputs["input_ids"]).long()
#             attention_mask = torch.tensor(inputs["attention_mask"]).long()
#             # token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
#             if labels is not None:
#                 labels = torch.tensor(inputs["labels"]).long()
#
#             # sample_input = (sample, input_ids, attention_mask, token_type_ids, labels)
#             sample_input = (sample, input_ids, attention_mask, labels)
#
#
#             all_inputs.append(sample_input)
#         return all_inputs
#
#     def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train",):
#         batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, data_type)
#         sample_list = []
#         input_ids_list = []
#         attention_mask_list = []
#         token_type_ids_list = []
#         labels_list = []
#
#         for sample in batch_data:
#             sample_list.append(sample[0])
#             input_ids_list.append(sample[1])
#             attention_mask_list.append(sample[2])
#             # token_type_ids_list.append(sample[3])
#
#             if data_type != "test":
#                 # labels_list.append(sample[4])
#                 labels_list.append(sample[3])
#
#         batch_input_ids = torch.stack(input_ids_list, dim=0)
#         batch_attention_mask = torch.stack(attention_mask_list, dim=0)
#         # batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
#         batch_labels = torch.stack(labels_list, dim=0) if data_type!="test" else None
#
#         return sample_list, batch_input_ids, batch_attention_mask, batch_labels
#         # return batch_token_type_ids,
#
#     def decode_ent(self, pred_matrix):
#         pass
#
#
# class MetricsCalculator:
#     def __init__(self):
#         super().__init__()
#
#     def get_sample_f1(self, y_pred, y_true):
#         y_pred = torch.gt(y_pred, 0).float()
#         return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)
#
#     def get_sample_precision(self, y_pred, y_true):
#         y_pred = torch.gt(y_pred, 0).float()
#         return torch.sum(y_pred[y_true == 1]) / (y_pred.sum()+1)
#
#     def get_evaluate_fpr(self, y_pred, y_true):
#         y_pred = y_pred.cpu().numpy()
#         y_true = y_true.cpu().numpy()
#         pred = []
#         true = []
#         for b, l, start, end in zip(*np.where(y_pred>0)):
#             pred.append((b, l, start, end))
#         for b, l, start, end in zip(*np.where(y_true>0)):
#             true.append((b, l, start, end))
#
#         R = set(pred)
#         T = set(true)
#         X = len(R & T)
#         Y = len(R)
#         Z = len(T)
#         f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
#         return R, T, X, Y, Z


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True, alibi=False):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE
        self.alibi = alibi

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)

        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def sinusoidal_position_embedding_1(self, batch_size, seq_len, output_dim):
        # position_enc = np.array(
        #     [
        #         [pos / np.power(10000, 2 * (j // 2) / output_dim) for j in range(output_dim)]
        #         for pos in range(seq_len)
        #     ]
        # )
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        indices = indices.repeat_interleave(2, dim=-1)
        embeddings = position_ids * indices
        output = torch.FloatTensor(embeddings)
        output[:, 0:output_dim // 2] = torch.sin(embeddings[:, 0::2])
        output[:, output_dim // 2:] = torch.cos(embeddings[:, 1::2])
        e_embeddings = output.repeat((batch_size, *([1] * len(output.shape)))).to(self.device)
        sinusoidal_pos = e_embeddings.chunk(2, dim=-1)
        return sinusoidal_pos

    def apply_rotary(self, x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        sin, cos = sin[..., None, :], cos[..., None, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, input_ids, attention_mask, test=True):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask,)
        # context_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[...,:self.inner_dim], outputs[...,self.inner_dim:] # TODO:修改为Linear获取？

        if self.RoPE and not test:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        if self.RoPE and test:
            sinusoidal_pos = self.sinusoidal_position_embedding_1(batch_size, seq_len, self.inner_dim)
            qw = self.apply_rotary(qw, sinusoidal_pos)
            kw = self.apply_rotary(kw, sinusoidal_pos)




        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits/self.inner_dim**0.5


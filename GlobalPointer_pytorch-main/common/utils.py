"""
Date: 2021-06-01 22:29:43
LastEditors: GodK
LastEditTime: 2021-07-31 19:30:18
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class DataMaker:
    def __init__(self, tokenizer, add_special_tokens=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
        """生成喂入模型的数据

        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            max_seq_len (int): 句子最大token数量
            ent2id (dict): ent到id的映射
            data_type (str, optional): data类型. Defaults to "train".

        Returns:
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
        """

        ent_type_size = len(ent2id)  # 实体类别

        all_inputs = []
        for sample in datas:

            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True
            )
            if 'chinese' not in self.tokenizer.name_or_path:
                language = 'english'
            else:
                language = 'chinese'

            labels = None
            if data_type != "test":
                ent2token_spans = self.preprocessor.get_ent2token_spans(
                    sample["text"], sample["entity_list"], language=language
                )
                labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                for start, end, label in ent2token_spans:
                    if start >= 512 or end >= 512:
                        continue
                        # print(start, end)
                    labels[ent2id[label], start, end] = 1
            inputs["labels"] = labels

            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            # token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            if labels is not None:
                labels = torch.tensor(inputs["labels"]).long()

            # sample_input = (sample, input_ids, attention_mask, token_type_ids, labels)
            sample_input = (sample, input_ids, attention_mask, labels)

            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train", ):
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, data_type)
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        labels_list = []

        for sample in batch_data:
            sample_list.append(sample[0])
            input_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            # token_type_ids_list.append(sample[3])

            if data_type != "test":
                # labels_list.append(sample[4])
                labels_list.append(sample[3])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        # batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_labels = torch.stack(labels_list, dim=0) if data_type != "test" else None

        return sample_list, batch_input_ids, batch_attention_mask, batch_labels
        # return batch_token_type_ids,

    def decode_ent(self, pred_matrix):
        pass


class MetricsCalculator:
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return R, T, X, Y, Z


class Preprocessor(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super(Preprocessor, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def get_ent2token_spans(self, text, entity_list, language='chinese'):
        """实体列表转为token_spans

        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """

        ent2token_spans = []
        text = text.strip()
        inputs = self.tokenizer(text, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)
        token2char_span_mapping = inputs["offset_mapping"]
        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)

        for ent_span in entity_list:
            if language == 'english':
                words = text.split()

                text2tokens_before = ' '.join(words[:ent_span[0]])
                text2tokens_ent = ' '.join(words[:ent_span[1]+1])
                ind_s = len(text2tokens_before) + 1 if text2tokens_before else 0
                ind_e = len(text2tokens_ent)
                start_span = None
                end_span = None
                for i, ind in enumerate(token2char_span_mapping):
                    if ind[0] == ind_s and start_span == None and ind != (0, 0):
                        start_span = i
                    if ind[1] == ind_e and end_span == None:
                        end_span = i
                    if start_span and end_span:
                        break

                # start_span_token = len(self.tokenizer.tokenize(text2tokens_before, add_special_tokens=False)) + 1
                # end_span_token = len(self.tokenizer.tokenize(text2tokens_ent, add_special_tokens=False))
                token_span = (start_span, end_span, ent_span[2])
                # assert start_span_token == start_span and end_span_token == end_span
                ent2token_spans.append(token_span)



            else:
                ent = text[ent_span[0]:ent_span[1] + 1]
                ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)


                # 寻找ent的token_span
                token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]
                token_end_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[-1]]

                token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[0], token_start_indexs))
                token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1] - 1 == ent_span[1], token_end_indexs))  # token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间

                if len(token_start_index) == 0 or len(token_end_index) == 0:
                    # print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                    continue
                token_span = (token_start_index[0], token_end_index[0], ent_span[2])
                ent2token_spans.append(token_span)

        return ent2token_spans


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()
import copy
from typing import Dict, Union, List
import numpy as np
from datasets import Dataset
from collections import defaultdict
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, TrainingArguments

from seed.utils.arguments import (
    TokenPairNERDataTrainingArguments,
    TokenPairEREDataTrainingArguments,
    TokenClassificationDataTrainingArguments,
    AutoFormatDataTrainingArguments,
    AutoFormatV2DataTrainingArguments
)


def get_entity_bio(seq: List, id2label: Dict[int, str]):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
        id2label (dict): map of label id to label str.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for idx, tag in enumerate(seq):
        if tag == -100:
            continue
        elif not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[0] = tag.split('-')[1]
            chunk[2] = idx
            if idx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = idx

            if idx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    if chunk[2] != -1:
        chunks.append(chunk)
    return chunks


class TokenPairNERDataOperator(object):
    def __init__(
            self,
            data_args: TokenPairNERDataTrainingArguments,
            tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast] = None
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.is_train = True
        self.remove_columns = [self.data_args.text_column_name, self.data_args.entity_column_name]

    def process(self, dataset: Dataset) -> Dataset:

        if self.data_args.length_enhancement:
            dataset = dataset.map(
                self.get_length,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Get length",
            )
            dataset = dataset.map(
                self.length_enhancement,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Length enhancement",
                remove_columns=["length"]
            )
        dataset = dataset.map(
            self.convert_char_label_to_token_label,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert char label to token label",
        )
        dataset = dataset.map(
            self.split_into_short_samples,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Split into short samples",
            remove_columns=self.remove_columns
        )
        dataset = dataset.map(
            self.padding_example_to_same_length,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Padding example to the same length",
        )

        return dataset

    def get_length(self, examples: Dict) -> Dict:
        text_column = self.data_args.text_column_name
        tokens = self.tokenizer(examples[text_column], return_length=True)
        return {"length": tokens['length']}

    def length_enhancement(self, examples: Dict) -> Dict:

        text_column = self.data_args.text_column_name
        label_column = self.data_args.entity_column_name
        min_length = np.min(examples["length"])

        new_examples = defaultdict(list)
        for i in range(len(examples[text_column])):
            text = examples[text_column][i]
            labels = examples[label_column][i]
            length = examples["length"][i]
            if length >= self.data_args.length_enhancement:
                new_examples[text_column].append(text)
                new_examples[label_column].append(labels)
            else:
                choice_num = int(self.data_args.length_enhancement / min_length) + 1
                data_ids = np.random.choice(list(range(len(examples[text_column]))), choice_num, replace=False)
                cum_length, n = length, 0
                new_labels = copy.deepcopy(labels)
                while cum_length < self.data_args.length_enhancement:
                    for entity in examples[label_column][data_ids[n]]:
                        new_entity = copy.deepcopy(entity)
                        new_entity['char_span'][0] += len(text)
                        new_entity['char_span'][1] += len(text)
                        new_labels.append(new_entity)
                    text = text + examples[text_column][data_ids[n]]
                    cum_length += examples["length"][data_ids[n]]
                    n += 1
                new_examples[text_column].append(text)
                new_examples[label_column].append(new_labels)

        return new_examples

    def convert_char_label_to_token_label(self, examples: Dict) -> Dict:

        text_column = self.data_args.text_column_name
        label_column = self.data_args.entity_column_name
        label2id = self.data_args.entity2id

        tokens = self.tokenizer(examples[text_column], add_special_tokens=self.data_args.add_special_tokens,
                                is_split_into_words=self.data_args.is_split_into_words,
                                return_offsets_mapping=True)

        token_labels = []
        for i in range(len(examples[text_column])):

            char_labels = examples[label_column][i]
            offset_mapping = tokens['offset_mapping'][i]
            token_label = []
            for entity in char_labels:
                start_char, end_char, label = entity['char_span'][0], entity['char_span'][1], label2id[entity['type']]
                start_token, end_token = None, None
                for n, (start, end) in enumerate(offset_mapping):
                    if start <= start_char < end:
                        # start_token = n + (1 if self.data_args.add_special_tokens else 0)
                        start_token = n
                        break
                for n, (start, end) in enumerate(offset_mapping):
                    if start < end_char <= end:
                        # end_token = n + (1 if self.data_args.add_special_tokens else 0)
                        end_token = n
                        break
                token_label.append((start_token, end_token, label))
            token_labels.append(token_label)

        return {label_column: token_labels}

    def split_into_short_samples(self, examples: Dict) -> Dict:

        if self.is_train:
            max_raw_seq_length = self.data_args.max_seq_length - (2 if self.data_args.add_special_tokens else 0)
            stride_window = self.data_args.stride_window
        else:
            max_raw_seq_length = 512 - (2 if self.data_args.add_special_tokens else 0)
            stride_window = 128

        new_examples = defaultdict(list)
        text_column, label_column = self.data_args.text_column_name, self.data_args.entity_column_name

        tokens = self.tokenizer(examples[text_column], add_special_tokens=False,
                                is_split_into_words=self.data_args.is_split_into_words,
                                return_offsets_mapping=True)

        for i in range(len(examples[text_column])):

            left_token, right_token, last_run = 0, max_raw_seq_length, False

            word_ids = tokens.word_ids(batch_index=i)
            sequence_length = len(word_ids)

            while not last_run:
                if right_token >= sequence_length + (2 if self.data_args.add_special_tokens else 0):
                    last_run = True
                while word_ids[left_token - 1] == word_ids[left_token]:
                    left_token -= 1
                    right_token = left_token + max_raw_seq_length

                new_examples["input_ids"].append(
                    self.tokenizer.build_inputs_with_special_tokens(tokens['input_ids'][i][left_token:right_token])
                    if self.data_args.add_special_tokens else tokens['input_ids'][i][left_token:right_token]
                )
                new_examples["attention_mask"].append(
                    [1] + tokens['attention_mask'][i][left_token:right_token] + [1]
                    if self.data_args.add_special_tokens else tokens['attention_mask'][i][left_token:right_token]
                )
                if "token_type_ids" in tokens:
                    new_examples["token_type_ids"].append(
                        self.tokenizer.create_token_type_ids_from_sequences(
                            tokens['token_type_ids'][i][left_token:right_token]
                        )
                        if self.data_args.add_special_tokens else tokens['token_type_ids'][i][left_token:right_token]
                    )
                new_examples["word_ids"].append(
                    [None] + word_ids[left_token:right_token] + [None]
                    if self.data_args.add_special_tokens else word_ids[left_token:right_token]
                )

                new_examples["labels"].append(
                    [(st - left_token, ed - left_token, t) for (st, ed, t) in examples[label_column][i]
                     if left_token <= st and right_token > ed]   # right token can not be equal to ed
                )
                new_examples["length"].append(len(new_examples["input_ids"][-1]))

                if last_run:
                    break
                left_token, right_token = left_token + stride_window, left_token + stride_window + max_raw_seq_length

        return new_examples

    def padding_example_to_same_length(self, examples: Dict) -> Dict:

        new_examples = defaultdict(list)
        seq_lengths, max_length = examples.pop('length'), self.data_args.max_seq_length if self.is_train else 512
        pad_token_id = self.tokenizer.pad_token_id

        for i in range(len(seq_lengths)):
            new_examples["input_ids"].append(examples['input_ids'][i] + (max_length - seq_lengths[i]) * [pad_token_id])
            new_examples["attention_mask"].append(examples['attention_mask'][i] + (max_length - seq_lengths[i]) * [0])
            if "token_type_ids" in examples:
                new_examples["token_type_ids"].append(examples['token_type_ids'][i] + (max_length - seq_lengths[i]) * [0])
            new_examples["word_ids"].append(examples['word_ids'][i] + (max_length - seq_lengths[i]) * [None])
            new_examples["labels"].append(examples['labels'][i])

        return new_examples

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


class TokenPairEREDataOperator(TokenPairNERDataOperator):
    def __init__(
            self,
            data_args: TokenPairEREDataTrainingArguments,
            tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast] = None,
    ):
        super().__init__(data_args, tokenizer)
        self.data_args = data_args
        self.remove_columns = [self.data_args.text_column_name, self.data_args.entity_column_name,
                               self.data_args.relation_column_name, 'jx']

    def convert_char_label_to_token_label(self, examples: Dict) -> Dict:

        text_column = self.data_args.text_column_name
        entity_column, relation_column = self.data_args.entity_column_name, self.data_args.relation_column_name
        label2id = self.data_args.label2id

        tokens = self.tokenizer(examples[text_column], add_special_tokens=self.data_args.add_special_tokens,
                                is_split_into_words=self.data_args.is_split_into_words,
                                return_offsets_mapping=True)

        entity_labels, relation_labels = list(), list()
        for i in range(len(examples[text_column])):
            entities = examples[entity_column][i]
            relations = examples[relation_column][i]
            offset_mapping = tokens['offset_mapping'][i]
            # start2token_mapping = {v[0]: k for k, v in enumerate(offset_mapping)}
            # end2token_mapping = {v[1]: k for k, v in enumerate(offset_mapping)}
            entity_label, relation_label = list(), list()
            start2token_mapping, end2token_mapping = dict(), dict()

            for entity in entities:
                start_char, end_char, label = entity['char_span'][0], entity['char_span'][1], label2id[entity['type']]
                # if start_char not in start2token_mapping or end_char not in end2token_mapping:
                #     print(1)
                #     continue
                # start_token, end_token = start2token_mapping[start_char], end2token_mapping[end_char]
                start_token, end_token = None, None
                for n, (start, end) in enumerate(offset_mapping):
                    if start <= start_char < end:
                        # start_token = n + (1 if self.data_args.add_special_tokens else 0)
                        start_token = n
                        start2token_mapping[start_char] = start_token
                        break
                for n, (start, end) in enumerate(offset_mapping):
                    if start < end_char <= end:
                        # end_token = n + (1 if self.data_args.add_special_tokens else 0)
                        end_token = n
                        end2token_mapping[end_char] = end_token
                        break
                if start_token is None or end_token is None:
                    print(1)
                entity_label.append((start_token, end_token, label))

            for relation in relations:
                relation_label.append(
                    self.get_token_label_from_relation(relation, start2token_mapping, end2token_mapping)
                )

            entity_labels.append(entity_label)
            relation_labels.append(relation_label)

        return {entity_column: entity_labels, relation_column: relation_labels}

    def get_token_label_from_relation(self, relation: Dict, start2token_mapping: Dict, end2token_mapping: Dict) -> List:

        label2id = self.data_args.label2id
        sub_start_char, sub_end_char = relation['sub_char_span'][0], relation['sub_char_span'][1]
        obj_start_char, obj_end_char = relation['obj_char_span'][0], relation['obj_char_span'][1]

        predicate = relation['predicate']
        temp_tuple = list()
        if self.data_args.directed_relation:
            if sub_start_char <= obj_start_char:
                temp_tuple.append(
                    (start2token_mapping[sub_start_char],
                     start2token_mapping[obj_start_char],
                     label2id[predicate + '#SH2OH'])
                )
            else:
                temp_tuple.append(
                    (start2token_mapping[obj_start_char],
                     start2token_mapping[sub_start_char],
                     label2id[predicate + '#OH2SH'])
                )
            if sub_end_char <= obj_end_char:
                temp_tuple.append(
                    (end2token_mapping[sub_end_char],
                     end2token_mapping[obj_end_char],
                     label2id[predicate + '#ST2OT'])
                )
            else:
                temp_tuple.append(
                    (end2token_mapping[sub_end_char],
                     end2token_mapping[obj_end_char],
                     label2id[predicate + '#ST2OT'])
                )
        else:
            if sub_start_char <= obj_start_char:
                temp_tuple.append(
                    (start2token_mapping[sub_start_char],
                     start2token_mapping[obj_start_char],
                     label2id[predicate + '#H2H'])
                )
            else:
                temp_tuple.append(
                    (start2token_mapping[obj_start_char],
                     start2token_mapping[sub_start_char],
                     label2id[predicate + '#H2H'])
                )
            if sub_end_char <= obj_end_char:
                temp_tuple.append(
                    (end2token_mapping[sub_end_char],
                     end2token_mapping[obj_end_char],
                     label2id[predicate + '#T2T'])
                )
            else:
                temp_tuple.append(
                    (end2token_mapping[obj_end_char],
                     end2token_mapping[sub_end_char],
                     label2id[predicate + '#T2T'])
                )

        return temp_tuple

    def split_into_short_samples(self, examples: Dict) -> Dict:

        if self.is_train:
            max_raw_seq_length = self.data_args.max_seq_length - (2 if self.data_args.add_special_tokens else 0)
            stride_window = self.data_args.stride_window
        else:
            max_raw_seq_length = 512 - (2 if self.data_args.add_special_tokens else 0)
            stride_window = 128

        new_examples = defaultdict(list)
        text_column = self.data_args.text_column_name
        entity_column, relation_column = self.data_args.entity_column_name, self.data_args.relation_column_name

        tokens = self.tokenizer(examples[text_column], add_special_tokens=False,
                                is_split_into_words=self.data_args.is_split_into_words,
                                return_offsets_mapping=True)

        for i in range(len(examples[text_column])):

            left_token, right_token, last_run = 0, max_raw_seq_length, False

            word_ids = tokens.word_ids(batch_index=i)
            sequence_length = len(word_ids)

            while not last_run:
                if right_token >= sequence_length + (2 if self.data_args.add_special_tokens else 0):
                    last_run = True
                while word_ids[left_token - 1] == word_ids[left_token] and 0 < left_token < sequence_length:
                    left_token -= 1
                    right_token = left_token + max_raw_seq_length

                new_examples["input_ids"].append(
                    self.tokenizer.build_inputs_with_special_tokens(tokens['input_ids'][i][left_token:right_token])
                    if self.data_args.add_special_tokens else tokens['input_ids'][i][left_token:right_token]
                )
                new_examples["attention_mask"].append(
                    [1] + tokens['attention_mask'][i][left_token:right_token] + [1]
                    if self.data_args.add_special_tokens else tokens['attention_mask'][i][left_token:right_token]
                )
                if "token_type_ids" in tokens:
                    new_examples["token_type_ids"].append(
                        self.tokenizer.create_token_type_ids_from_sequences(
                            tokens['token_type_ids'][i][left_token:right_token]
                        )
                        if self.data_args.add_special_tokens else tokens['token_type_ids'][i][left_token:right_token]
                    )
                new_examples["word_ids"].append(
                    [None] + word_ids[left_token:right_token] + [None]
                    if self.data_args.add_special_tokens else word_ids[left_token:right_token]
                )

                labels = [(st - left_token, ed - left_token, t) for (st, ed, t) in examples[entity_column][i]
                          if left_token <= st and right_token > ed]   # right token can not be equal to ed
                for temp_tuple in examples[relation_column][i]:
                    if left_token <= temp_tuple[0][0] and right_token > temp_tuple[0][1] and \
                            left_token <= temp_tuple[1][0] and right_token > temp_tuple[1][1]:  # match all conditions
                        head_tuple = (temp_tuple[0][0] - left_token, temp_tuple[0][1] - left_token, temp_tuple[0][-1])
                        labels.append(head_tuple)
                        tail_tuple = (temp_tuple[1][0] - left_token, temp_tuple[1][1] - left_token, temp_tuple[1][-1])
                        labels.append(tail_tuple)

                new_examples["labels"].append(labels)

                new_examples["length"].append(len(new_examples["input_ids"][-1]))

                if last_run:
                    break
                left_token, right_token = left_token + stride_window, left_token + stride_window + max_raw_seq_length

        return new_examples


class TokenClassificationDataOperator(object):
    def __init__(
            self,
            data_args: TokenClassificationDataTrainingArguments,
            tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast] = None,
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer

        self.start_token = self.tokenizer.cls_token
        self.end_token = self.tokenizer.sep_token
        self.is_train = True

    def process(self, dataset: Dataset) -> Dataset:

        dataset = dataset.map(
            self.convert_char_label_to_token_label,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert char label to token label",
        )
        # remove too long samples
        # dataset = dataset.filter(
        #     lambda example: len(self.tokenizer(example[self.data_args.text_column_name])["input_ids"]) <= 1024,
        #     num_proc=self.data_args.preprocessing_num_workers,
        #     load_from_cache_file=not self.data_args.overwrite_cache,
        #     desc="Filter too long samples",
        # )
        dataset = dataset.map(
            self.split_into_short_samples,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Split into short samples",
            remove_columns=dataset.column_names,
        )
        dataset = dataset.map(
            self.convert_sample_into_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert sample into features",
            remove_columns=[self.data_args.entity_column_name],
        )

        return dataset

    def convert_char_label_to_token_label(self, examples: Dict) -> Dict:

        text_column, label_column = self.data_args.text_column_name, self.data_args.entity_column_name
        label2id = self.data_args.entity2id

        tokens = self.tokenizer(examples[text_column], add_special_tokens=False,
                                is_split_into_words=self.data_args.is_split_into_words,
                                return_offsets_mapping=True)

        token_labels = []
        for i in range(len(examples[text_column])):

            entity_labels = examples[label_column][i]
            offset_mapping = tokens['offset_mapping'][i]
            # start2token_mapping = {v[0]: k for k, v in enumerate(offset_mapping)}
            # end2token_mapping = {v[1]: k for k, v in enumerate(offset_mapping)}
            token_label = []

            for entity in entity_labels:
                start_char, end_char, label = entity['char_span'][0], entity['char_span'][1], label2id[entity['type']]
                start_token, end_token = None, None
                for n, (start, end) in enumerate(offset_mapping):
                    if start <= start_char < end:
                        start_token = n + (1 if self.data_args.add_special_tokens else 0)
                        break
                for n, (start, end) in enumerate(offset_mapping):
                    if start < end_char <= end:
                        end_token = n + (2 if self.data_args.add_special_tokens else 1)
                        break
                # if start_token is None or end_token is None:
                #     print(1)
                # start_token = start2token_mapping[start_char] + 1 if self.data_args.add_special_tokens else 0
                # end_token = end2token_mapping[end_char] + 2 if self.data_args.add_special_tokens else 1
                token_label.append((start_token, end_token, label))

            token_labels.append(token_label)

        return {label_column: token_labels}

    def split_into_short_samples(self, examples: Dict) -> Dict:

        if self.is_train:
            max_raw_seq_length = self.data_args.max_seq_length - (2 if self.data_args.add_special_tokens else 0)
            stride_window = self.data_args.stride_window
        else:
            max_raw_seq_length = 512 - (2 if self.data_args.add_special_tokens else 0)
            stride_window = 128

        new_examples = defaultdict(list)
        text_column, label_column = self.data_args.text_column_name, self.data_args.entity_column_name
        # cls_token_id, eos_token_id = self.tokenizer.cls_token_id, self.tokenizer.eos_token_id

        tokens = self.tokenizer(examples[text_column], add_special_tokens=False,
                                is_split_into_words=self.data_args.is_split_into_words,
                                return_offsets_mapping=True)

        for i in range(len(examples[text_column])):

            left_token, right_token, last_run = 0, max_raw_seq_length, False
            word_ids = tokens.word_ids(batch_index=i)
            sequence_length = len(word_ids)

            while not last_run:
                if right_token >= sequence_length:  # + (2 if self.data_args.add_special_tokens else 0):
                    last_run = True
                while word_ids[left_token - 1] == word_ids[left_token] and 0 < left_token < sequence_length:
                    left_token -= 1
                    right_token = left_token + max_raw_seq_length

                new_examples["input_ids"].append(
                    self.tokenizer.build_inputs_with_special_tokens(tokens['input_ids'][i][left_token:right_token])
                    if self.data_args.add_special_tokens else tokens['input_ids'][i][left_token:right_token]
                )
                new_examples["attention_mask"].append(
                    [1] + tokens['attention_mask'][i][left_token:right_token] + [1]
                    if self.data_args.add_special_tokens else tokens['attention_mask'][i][left_token:right_token]
                )
                if "token_type_ids" in tokens:
                    new_examples["token_type_ids"].append(
                        self.tokenizer.create_token_type_ids_from_sequences(
                            tokens['token_type_ids'][i][left_token:right_token]
                        )
                        if self.data_args.add_special_tokens else tokens['token_type_ids'][i][left_token:right_token]
                    )
                new_examples["word_ids"].append(
                    [None] + word_ids[left_token:right_token] + [None]
                    if self.data_args.add_special_tokens else word_ids[left_token:right_token]
                )

                new_examples[label_column].append(
                    [(st - left_token, ed - left_token, t) for (st, ed, t) in examples[label_column][i]
                     if left_token <= st and right_token > ed]   # right token can not be equal to ed
                )
                if last_run:
                    break
                left_token, right_token = left_token + stride_window, left_token + stride_window + max_raw_seq_length

        return new_examples

    def convert_sample_into_features(self, examples: Dict) -> Dict:

        labels = examples.pop(self.data_args.entity_column_name)
        word_ids = examples.pop("word_ids")

        new_examples = defaultdict(list)
        for i in range(len(labels)):
            for key, value in examples.items():
                new_examples[key].append(value[i])
            word_id = word_ids[i]
            label = labels[i]
            new_examples['labels'].append(self.convert_token_label_to_sequence_label(word_id, label))

        return new_examples

    def convert_token_label_to_sequence_label(self, word_ids: List, label: List) -> List:
        sequence_label = []
        s_flag, previous_word_idx = 1, None
        for i in range(len(word_ids)):
            o_flag = 1
            if word_ids[i] is None:
                sequence_label.append(-100)
            elif word_ids[i] == previous_word_idx:
                sequence_label.append(-100)
            else:
                for start_token, end_token, entity_type in label:
                    if i == start_token:
                        sequence_label.append(
                            self.data_args.tag2id[f"B-{self.data_args.entity_types[entity_type]}"]
                        )
                        o_flag = 0
                        break
                    elif start_token < i < end_token:
                        sequence_label.append(
                            self.data_args.tag2id[f"I-{self.data_args.entity_types[entity_type]}"]
                        )
                        o_flag = 0
                        break
                if o_flag:
                    sequence_label.append(self.data_args.tag2id["O"])
            previous_word_idx = word_ids[i]
        return sequence_label

    # def convert_token_label_to_sequence_label(self, word_ids: List, label: List) -> List:
    #     sequence_label = []
    #     s_flag, previous_word_idx = 1, None
    #     for i in range(len(word_ids)):
    #         o_flag = 1
    #         if word_ids[i] is None:
    #             sequence_label.append(-100)
    #         elif word_ids[i] == previous_word_idx:
    #             sequence_label.append(-100)
    #         else:
    #             for start_token, end_token, entity_type in label:
    #                 if i == start_token:
    #                     sequence_label.append(
    #                         self.data_args.tag2id[f"B-{self.data_args.entity_types[entity_type]}"]
    #                     )
    #                     o_flag, s_flag = 0, 1
    #                     break
    #                 elif start_token < i < end_token:
    #                     sequence_label.append(
    #                         self.data_args.tag2id[f"I-{self.data_args.entity_types[entity_type]}"]
    #                     )
    #                     o_flag, s_flag = 0, 1
    #                     break
    #             if o_flag and s_flag:
    #                 sequence_label.append(self.data_args.label2id["B-content"])
    #                 s_flag = 0
    #             elif o_flag and not s_flag:
    #                 sequence_label.append(self.data_args.label2id["I-content"])
    #         previous_word_idx = word_ids[i]
    #     return sequence_label

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


class TokenClassificationV2DataOperator(object):
    def __init__(
            self,
            data_args: TokenClassificationDataTrainingArguments,
            tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast] = None,
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.is_train = True

    def process(self, dataset: Dataset) -> Dataset:

        dataset = dataset.map(
            self.convert_char_label_to_token_label,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert char label to token label",
        )
        # remove too long samples
        dataset = dataset.filter(
            lambda example: len(self.tokenizer(example[self.data_args.text_column_name])["input_ids"]) <= 1024,
        )
        dataset = dataset.map(
            self.split_into_short_samples,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Split into short samples",
            remove_columns=dataset.column_names,
        )
        dataset = dataset.map(
            self.convert_sample_into_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert sample into features",
            remove_columns=[self.data_args.label_column_name],
        )

        return dataset

    def convert_char_label_to_token_label(self, examples: Dict) -> Dict:

        text_column, label_column = self.data_args.text_column_name, self.data_args.label_column_name
        label2id = self.data_args.origin_label2id

        tokens = self.tokenizer(examples[text_column], add_special_tokens=False,
                                is_split_into_words=self.data_args.is_split_into_words,
                                return_offsets_mapping=True)

        token_labels = []
        for i in range(len(examples[text_column])):

            entity_labels = examples[label_column][i]
            offset_mapping = tokens['offset_mapping'][i]
            start2token_mapping = {v[0]: k for k, v in enumerate(offset_mapping)}
            end2token_mapping = {v[1]: k for k, v in enumerate(offset_mapping)}
            token_label = []

            for entity in entity_labels:
                start_char, end_char, label = entity['char_span'][0], entity['char_span'][1], label2id[entity['type']]
                start_token, end_token = start2token_mapping[start_char], end2token_mapping[end_char] + 1
                token_label.append((start_token, end_token, label))

            token_labels.append(token_label)

        return {label_column: token_labels}

    def split_into_short_samples(self, examples: Dict) -> Dict:

        if self.is_train:
            max_raw_seq_length = self.data_args.max_seq_length - (2 if self.data_args.add_special_tokens else 0)
            stride_window = self.data_args.stride_window
        else:
            max_raw_seq_length = 512 - (2 if self.data_args.add_special_tokens else 0)
            stride_window = 128

        new_examples = defaultdict(list)
        text_column, label_column = self.data_args.text_column_name, self.data_args.label_column_name
        cls_token_id, eos_token_id = self.tokenizer.cls_token_id, self.tokenizer.eos_token_id

        tokens = self.tokenizer(examples[text_column], add_special_tokens=False,
                                is_split_into_words=self.data_args.is_split_into_words,
                                return_offsets_mapping=True)

        for i in range(len(examples[text_column])):

            left_token, right_token, last_run = 0, max_raw_seq_length, False
            word_ids = tokens.word_ids(batch_index=i)
            sequence_length = len(word_ids)

            while not last_run:
                if right_token >= sequence_length:  # + (2 if self.data_args.add_special_tokens else 0):
                    last_run = True
                while word_ids[left_token - 1] == word_ids[left_token] and 0 < left_token < sequence_length:
                    left_token -= 1
                    right_token = left_token + max_raw_seq_length

                new_examples["input_ids"].append(
                    [cls_token_id] + tokens['input_ids'][i][left_token:right_token] + [eos_token_id]
                    if self.data_args.add_special_tokens else tokens['input_ids'][i][left_token:right_token]
                )
                new_examples["attention_mask"].append(
                    [1] + tokens['attention_mask'][i][left_token:right_token] + [1]
                    if self.data_args.add_special_tokens else tokens['attention_mask'][i][left_token:right_token]
                )
                if "token_type_ids" in tokens:
                    new_examples["token_type_ids"].append(
                        [0] + tokens['token_type_ids'][i][left_token:right_token] + [0]
                        if self.data_args.add_special_tokens else tokens['token_type_ids'][i][left_token:right_token]
                    )
                new_examples["word_ids"].append(
                    [None] + word_ids[left_token:right_token] + [None]
                    if self.data_args.add_special_tokens else word_ids[left_token:right_token]
                )

                new_examples[label_column].append(
                    [(max(st - left_token, 0), min(ed - left_token, max_raw_seq_length), t) for (st, ed, t) in examples[label_column][i]
                     if left_token <= st < right_token or left_token <= ed < right_token]   # right token can not be equal to ed
                )
                if last_run:
                    break
                left_token, right_token = left_token + stride_window, left_token + stride_window + max_raw_seq_length

        return new_examples

    def convert_sample_into_features(self, examples: Dict) -> Dict:

        labels = examples.pop(self.data_args.label_column_name)
        word_ids = examples.pop("word_ids")

        new_examples = defaultdict(list)
        for i in range(len(labels)):
            for key, value in examples.items():
                new_examples[key].append(value[i])
            word_id = word_ids[i]
            label = labels[i]
            new_examples['labels'].append(self.convert_token_label_to_sequence_label(word_id, label))

        return new_examples

    def convert_token_label_to_sequence_label(self, word_ids: List, label: List) -> List:
        sequence_label = []
        s_flag, previous_word_idx = 1, None
        for i in range(len(word_ids)):
            o_flag = 1
            if word_ids[i] is None:
                sequence_label.append(-100)
            elif word_ids[i] == previous_word_idx:
                sequence_label.append(-100)
            else:
                for start_token, end_token, entity_type in label:
                    if i == start_token:
                        sequence_label.append(
                            self.data_args.label2id[f"B-{self.data_args.id2origin_label[entity_type]}"]
                        )
                        o_flag, s_flag = 0, 1
                        break
                    elif start_token < i < end_token:
                        sequence_label.append(
                            self.data_args.label2id[f"I-{self.data_args.id2origin_label[entity_type]}"]
                        )
                        o_flag, s_flag = 0, 1
                        break
                if o_flag and s_flag:
                    sequence_label.append(self.data_args.label2id["B-content"])
                    s_flag = 0
                elif o_flag and not s_flag:
                    sequence_label.append(self.data_args.label2id["I-content"])
            previous_word_idx = word_ids[i]
        return sequence_label

    def convert_token_label_to_span_label(self, word_ids: List, label: List) -> List:
        sequence_label = []
        s_flag, previous_word_idx = 1, None
        for i in range(len(word_ids)):
            o_flag = 1
            if word_ids[i] is None:
                sequence_label.append(-100)
            elif word_ids[i] == previous_word_idx:
                sequence_label.append(-100)
            else:
                for start_token, end_token, entity_type in label:
                    if i == start_token:
                        sequence_label.append(
                            self.data_args.span_label2id[f"{self.data_args.id2origin_label[entity_type]}-start"]
                        )
                        o_flag, s_flag = 0, 1
                        break
                    elif start_token < i < end_token:
                        sequence_label.append(
                            self.data_args.label2id[f"I-{self.data_args.id2origin_label[entity_type]}"]
                        )
                        o_flag, s_flag = 0, 1
                        break
                if o_flag and s_flag:
                    sequence_label.append(self.data_args.label2id["B-content"])
                    s_flag = 0
                elif o_flag and not s_flag:
                    sequence_label.append(self.data_args.label2id["I-content"])
            previous_word_idx = word_ids[i]
        return sequence_label

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


class AutoFormatDataOperator(object):
    def __init__(
            self,
            data_args: TrainingArguments,
            tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast] = None,
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.is_train = True

    def process(self, dataset: Dataset) -> Dataset:

        dataset = dataset.filter(
            lambda x: 10 <= len(x[self.data_args.entity_column_name]) <= 512,
            desc="Filtering out too long entities"
        )
        dataset = dataset.map(
            self.convert_relation_into_labels,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert relation to labels",
        )
        dataset = dataset.map(
            self.convert_sample_into_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert sample into features",
            remove_columns=[
                self.data_args.text_column_name,
                self.data_args.entity_column_name,
                self.data_args.relation_column_name,
                "id",
                "au_flag",
                "link_flag"
            ],
        )

        return dataset

    def convert_relation_into_labels(self, examples: Dict) -> Dict:

        new_examples = defaultdict(list)
        text_column_name = self.data_args.text_column_name
        entity_list = examples[self.data_args.entity_column_name]
        relation_list = examples[self.data_args.relation_column_name]

        for i in range(len(examples[text_column_name])):

            entity_list[i].sort(key=lambda x: x["char_span"][0])
            new_examples["au_flag"].append([1 if entity["au"] else 0 for entity in entity_list[i]])
            start_char2id = {entity["char_span"][0]: i for i, entity in enumerate(entity_list[i])}

            link2sequence_id = self.convert_sequence_length_to_link2sequence_id(len(entity_list[i]))
            link_flag = [0] * len(link2sequence_id)
            for relation in relation_list[i]:
                sub, obj = relation["sub_char_span"][0], relation["obj_char_span"][0]
                link_flag[link2sequence_id[(start_char2id[sub], start_char2id[obj])]] = 1
            new_examples["link_flag"].append(link_flag)

        return new_examples

    @staticmethod
    def convert_sequence_length_to_link2sequence_id(sequence_length: int) -> Dict:
        link = [(i, j) for i in range(sequence_length) for j in range(i + 1, sequence_length)]
        return {l: i for i, l in enumerate(link)}

    def convert_sample_into_features(self, examples: Dict) -> Dict:

        new_examples = defaultdict(list)
        entity_column_name = self.data_args.entity_column_name
        for i in range(len(examples[entity_column_name])):
            examples[entity_column_name][i].sort(key=lambda x: x["char_span"][0])
            entities = list()
            for entity in examples[entity_column_name][i]:
                entity['text'] = f"<{entity['type']}>" + entity['text'] + f"</{entity['type']}>"
                entities.append(entity['text'])
            tokens = self.tokenizer(entities, add_special_tokens=False, padding="max_length", truncation=True,
                                    max_length=128)
            for key, value in tokens.items():
                new_examples[key].append(value)
        new_examples["node_labels"] = examples["au_flag"]
        new_examples["link_labels"] = examples["link_flag"]

        return new_examples

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


class AutoFormatV2DataOperator(object):
    def __init__(
            self,
            data_args: AutoFormatV2DataTrainingArguments,
            tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast] = None,
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.is_train = True

    def process(self, dataset: Dataset) -> Dataset:

        dataset = dataset.filter(
            lambda x: 2 <= len(x[self.data_args.entity_column_name]) <= 512,
            num_proc=self.data_args.preprocessing_num_workers,
            desc="Filtering out too long entities"
        )
        dataset = dataset.map(
            self.convert_relation_into_labels,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert relation to labels",
        )
        dataset = dataset.map(
            self.convert_sample_into_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Convert sample into features",
            remove_columns=[
                self.data_args.text_column_name,
                self.data_args.entity_column_name,
                self.data_args.relation_column_name,
                "id"
            ],
        )

        return dataset

    def convert_relation_into_labels(self, examples: Dict) -> Dict:

        new_examples = defaultdict(list)

        entity_list = examples[self.data_args.entity_column_name]
        relation_list = examples[self.data_args.relation_column_name]

        for i in range(len(examples[self.data_args.text_column_name])):

            entity_list[i].sort(key=lambda x: x["char_span"][0])
            new_examples["node_labels"].append([self.data_args.label2id[entity['type']] for entity in entity_list[i]])
            new_examples["au_labels"].append([1 if entity["au"] else 0 for entity in entity_list[i]])
            start_char2id = {entity["char_span"][0]: i for i, entity in enumerate(entity_list[i])}

            link_flag = list(range(len(entity_list[i])))
            for relation in relation_list[i]:
                sub, obj = relation["sub_char_span"][0], relation["obj_char_span"][0]
                link_flag[start_char2id[obj]] = start_char2id[sub]
            new_examples["link_labels"].append(link_flag)

        return new_examples

    def convert_sample_into_features(self, examples: Dict) -> Dict:

        new_examples = defaultdict(list)
        entity_column_name = self.data_args.entity_column_name
        for i in range(len(examples[entity_column_name])):
            examples[entity_column_name][i].sort(key=lambda x: x["char_span"][0])
            entities = list()
            for entity in examples[entity_column_name][i]:
                entity['text'] = f"<{entity['type']}>" + entity['text'] + f"</{entity['type']}>"
                entities.append(entity['text'])
            tokens = self.tokenizer(entities, add_special_tokens=False, padding="max_length", truncation=True,
                                    max_length=128)
            for key, value in tokens.items():
                new_examples[key].append(value)

        return new_examples

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


# class AutoFormatV3DataOperator(object):
    # def __init__(
    #         self,
    #         data_args: AutoFormatV2DataTrainingArguments,
    #         tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast] = None,
    # ):
    #     self.data_args = data_args
    #     self.tokenizer = tokenizer
    #     self.is_train = True
    #
    # def process(self, dataset: Dataset) -> Dataset:
    #
    #     dataset = dataset.filter(
    #         lambda x: 2 <= len(x[self.data_args.entity_column_name]) <= 512,
    #         desc="Filtering out too long entities"
    #     )
    #     dataset = dataset.map(
    #         self.convert_relation_into_labels,
    #         batched=True,
    #         num_proc=self.data_args.preprocessing_num_workers,
    #         load_from_cache_file=not self.data_args.overwrite_cache,
    #         desc="Convert relation to labels",
    #     )
    #     dataset = dataset.map(
    #         self.convert_sample_into_features,
    #         batched=True,
    #         num_proc=self.data_args.preprocessing_num_workers,
    #         load_from_cache_file=not self.data_args.overwrite_cache,
    #         desc="Convert sample into features",
    #         remove_columns=[
    #             self.data_args.text_column_name,
    #             self.data_args.entity_column_name,
    #             self.data_args.relation_column_name,
    #             "id"
    #         ],
    #     )
    #
    #     return dataset
    #
    # def convert_relation_into_labels(self, examples: Dict) -> Dict:
    #
    #     new_examples = defaultdict(list)
    #
    #     entity_list = examples[self.data_args.entity_column_name]
    #     relation_list = examples[self.data_args.relation_column_name]
    #
    #     for i in range(len(examples[self.data_args.text_column_name])):
    #
    #         entity_list[i].sort(key=lambda x: x["char_span"][0])
    #         new_examples["node_labels"].append([self.data_args.label2id[entity['type']] for entity in entity_list[i]])
    #         new_examples["au_labels"].append([1 if entity["au"] else 0 for entity in entity_list[i]])
    #         start_char2id = {entity["char_span"][0]: i for i, entity in enumerate(entity_list[i])}
    #
    #         link2sequence_id = self.convert_sequence_length_to_link2sequence_id(len(entity_list[i]))
    #         link_flag = [0] * len(link2sequence_id)
    #         for relation in relation_list[i]:
    #             sub, obj = relation["sub_char_span"][0], relation["obj_char_span"][0]
    #             link_flag[link2sequence_id[(start_char2id[sub], start_char2id[obj])]] = 1
    #         new_examples["link_labels"].append(link_flag)
    #
    #     return new_examples
    #
    # @staticmethod
    # def convert_sequence_length_to_link2sequence_id(sequence_length: int) -> Dict:
    #     link = [(i, j) for i in range(sequence_length) for j in range(i + 1, sequence_length)]
    #     return {l: i for i, l in enumerate(link)}
    #
    # def convert_sample_into_features(self, examples: Dict) -> Dict:
    #
    #     new_examples = defaultdict(list)
    #     entity_column_name = self.data_args.entity_column_name
    #     for i in range(len(examples[entity_column_name])):
    #         examples[entity_column_name][i].sort(key=lambda x: x["char_span"][0])
    #         entities = list()
    #         for entity in examples[entity_column_name][i]:
    #             entity['text'] = f"<{entity['type']}>" + entity['text'] + f"</{entity['type']}>"
    #             entities.append(entity['text'])
    #         tokens = self.tokenizer(entities, add_special_tokens=False, padding="max_length", truncation=True,
    #                                 max_length=128)
    #         for key, value in tokens.items():
    #             new_examples[key].append(value)
    #
    #     return new_examples
    #
    # def train(self):
    #     self.is_train = True
    #
    # def eval(self):
    #     self.is_train = False


import regex as re
from dataclasses import dataclass

import torch
import logging
from collections import defaultdict
import itertools
from typing import Optional, Union, Dict
import numpy as np
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, PreTrainedModel
from transformers.file_utils import PaddingStrategy
from datasets import concatenate_datasets


logger = logging.getLogger("__main__")


# class DynamicNERPromptGenerator:
#     """
#     Sample negative prompt
#     Todo: over sample high-similarity prompts
#     """
#     def __init__(self, prompts, cascade_flag=True, cascade_mapping=None, up_sample_rate=1):
#         self.prompts = prompts
#         if cascade_flag and not cascade_mapping:
#             cascade_mapping = self._generate_cascade_mapping(prompts)
#         self.cascade_mapping = cascade_mapping
#         self.up_sample_rate = up_sample_rate
#         if cascade_flag:
#             self.sample_negative_prompt = self._sample_negative_prompt_with_cascade
#         else:
#             self.sample_negative_prompt = self._sample_negative_prompt
#         logger.info(f"Meta Sample, Cascade Flag: {cascade_flag}, Negative Rate: {self.up_sample_rate}")
#
#     @staticmethod
#     def _generate_cascade_mapping(prompts):
#         coarse2fine = defaultdict(set)
#         for prompt in prompts:
#             coarse2fine[prompt.split("-")[0]].add(prompt)
#         return dict(coarse2fine)
#
#     def _sample_negative_prompt(self, positive_prompts):
#         """ Sample prompts
#         """
#         negative_prompts = set(self.prompts) - set(positive_prompts)
#         negative_prompt = np.random.choice(list(negative_prompts))
#
#         return negative_prompt
#
#     def _sample_negative_prompt_with_cascade(self, positive_prompts):
#         """ Sample prompts
#         """
#         negative_prompts = set(self.prompts) - set(positive_prompts)
#         up_sample_prompts = {p for prompt in negative_prompts for p in self.cascade_mapping[prompt.split("-")[0]]}
#         up_sample_prompts = up_sample_prompts - set(positive_prompts)
#         p = np.array([1 + (self.up_sample_rate if prompt in up_sample_prompts else 0) for prompt in negative_prompts])
#         p = p / sum(p)
#         negative_prompt = np.random.choice(list(negative_prompts), p=p)
#
#         return negative_prompt


@dataclass
class DataCollatorForTokenPairClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: Optional[Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]] = None
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: int = 128
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    num_of_tags: int = 0
    matrix_to_token_pair: Dict = None
    token_pair_length: int = 8256
    training: bool = True

    def __post_init__(self):
        self.token_pair_length = (self.max_length + 1) * self.max_length // 2
        matrix_sequence = [(i, j) for i in range(self.max_length) for j in range(i, self.max_length)]
        self.matrix_to_token_pair = {tp:n for n, tp in enumerate(matrix_sequence)}

        self.train_max_length = self.max_length
        self.train_token_pair_length = self.token_pair_length
        self.train_matrix_to_token_pair = self.matrix_to_token_pair

        self.eval_max_length = 512
        self.eval_token_pair_length = (self.eval_max_length + 1) * self.eval_max_length // 2
        eval_matrix_sequence = [(i, j) for i in range(self.eval_max_length) for j in range(i, self.eval_max_length)]
        self.eval_matrix_to_token_pair = {tp:n for n, tp in enumerate(eval_matrix_sequence)}
        self.token_pair_to_matrix = {n:tp for tp, n in self.eval_matrix_to_token_pair.items()}

    def __call__(self, features):
        """ Make Meta Schema Batch

        Args:
            features (Dict): [description]
                - sample_prompt: indicates sample_prompt example, need pop after call
                - spots (List[str]): List of spots in this sentence, need pop after call
                - asocs (List[str]): List of asocs in this sentence, need pop after call
                - input_ids
                - attention_mask
                - labels

        Returns:
        """

        output_features = list()
        for n, feature in enumerate(features):

            # new_feature = copy.deepcopy(feature)
            labels = feature.pop("labels")
            word_ids = feature.pop("word_ids")
            new_labels = np.zeros((self.token_pair_length, self.num_of_tags), dtype=np.long)
            shaking_tag_mask = self.word_ids2token_pair_mask(word_ids)
            new_labels = np.where(shaking_tag_mask[:,np.newaxis], -100, new_labels)
            for label in labels:
                new_labels[self.matrix_to_token_pair[tuple(label[:2])], label[2]] = 1
            feature["labels"] = new_labels.tolist()
            output_features.append(feature)

        output_features = self.tokenizer.pad(
            output_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        return output_features

    @staticmethod
    def word_ids2token_pair_mask(word_ids):
        """
        mask shaking_tag that belongs to sub-word
        :param word_ids:
        :return:
        """
        previous_word_idx = None
        mask = list()
        for word_idx in word_ids:
            mask.append(0 if word_idx != previous_word_idx else 1)
            previous_word_idx = word_idx
        # 这里mask要向前走一步, 取mask[-1]是为了防止最后一位是att_mask
        mask = mask[1:] + [mask[-1]]
        seq_mask = list()
        for i in range(len(mask)):
            seq_mask.append(mask[i:])
        token_pair_mask = list(itertools.chain(*seq_mask))
        token_pair_mask = np.array(token_pair_mask)
        return token_pair_mask

    def train(self):
        self.training = True
        self.max_length = self.train_max_length
        self.token_pair_length = self.train_token_pair_length
        self.matrix_to_token_pair = self.train_matrix_to_token_pair

    def eval(self):
        self.training = False
        self.max_length = self.eval_max_length
        self.token_pair_length = self.eval_token_pair_length
        self.matrix_to_token_pair = self.eval_matrix_to_token_pair


# @dataclass
# class DataCollatorForUniversalNER:
#     """
#     Data collator that will dynamically pad the inputs received, as well as the labels.
#
#     Args:
#         tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
#             The tokenizer used for encoding the data.
#         model (:class:`~transformers.PreTrainedModel`):
#             The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
#             prepare the `decoder_input_ids`
#
#             This is useful when using `label_smoothing` to avoid calculating loss twice.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:
#
#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence is provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the returned list and optionally padding length (see above).
#         max_target_length (:obj:`int`, `optional`):
#             Maximum length of target sequence length.
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.
#
#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#         label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
#             The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
#     """
#
#     tokenizer: PreTrainedTokenizerFast
#     negative_sampler: DynamicNERPromptGenerator
#     model: Optional[PreTrainedModel] = None
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     max_target_length: Optional[int] = None
#     max_prefix_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     label_pad_token_id: int = -100
#
#     def __call__(self, features):
#         """ Make Meta Schema Batch
#
#         Args:
#             features (Dict): [description]
#                 - sample_prompt: indicates sample_prompt example, need pop after call
#                 - spots (List[str]): List of spots in this sentence, need pop after call
#                 - asocs (List[str]): List of asocs in this sentence, need pop after call
#                 - input_ids
#                 - attention_mask
#                 - labels
#
#         Returns:
#         """
#         output_features = list()
#         for feature in features:
#
#             if feature['enable_sample']:
#                 negative_prompt = self.negative_sampler.sample_negative_prompt(positive_prompts=feature.get('prompts'))
#             else:
#                 negative_prompt = feature.get('prompts')[0]
#             prompt = re.split(r"/|-", negative_prompt)
#             input_text = feature.get('words')
#             tokens = self.tokenizer(prompt, input_text,
#                                     padding="max_length", truncation="only_second",
#                                     max_length=128, is_split_into_words=True)
#             label = feature.get('labels')
#
#             prompt_len = len(self.tokenizer(prompt, is_split_into_words=True)['input_ids'])
#             # start_label, end_label = [-100] * prompt_len, [-100] * prompt_len
#             # previous_word_idx = None
#             tokens["start_labels"], tokens["end_labels"] = self.generate_start_and_end_labels(tokens.word_ids(),
#                                                                                               tokens.sequence_ids(),
#                                                                                               label)
#             output_features.append(tokens)
#
#         output_features = self.padding_features_and_labels(output_features)
#
#         return output_features
#
#     @staticmethod
#     def generate_start_and_end_labels(word_ids, sequence_ids, label):
#         start_label, end_label = list(), list()
#         start_ids, end_ids = {l[0] for l in label}, {l[1] - 1 for l in label}
#         for idx, s_idx in zip(word_ids, sequence_ids):
#             if s_idx != 1:
#                 start_label.append(-100)
#                 end_label.append(-100)
#                 previous_idx = idx
#             elif idx == previous_idx or idx is None:
#                 start_label.append(-100)
#                 end_label.append(-100)
#             elif idx != previous_idx:
#                 if idx in start_ids:
#                     start_label.append(1)
#                 else:
#                     start_label.append(0)
#                 if idx in end_ids:
#                     end_label.append(1)
#                 else:
#                     end_label.append(0)
#                 previous_idx = idx
#         return start_label, end_label
#
#     def padding_features_and_labels(self, features):
#
#         label_names = [key for key in features[0].keys() if "label" in key]
#         labels = {label_name:[feature[label_name] for feature in features] for label_name in label_names}
#         batch = self.tokenizer.pad(
#             features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             # Conversion to tensors will fail if we have labels as they are not of the same length yet.
#             return_tensors="pt" if label_names is None else None,
#         )
#
#         if label_names is None:
#             return batch
#
#         sequence_length = torch.tensor(batch["input_ids"]).shape[1]
#         # print(sequence_length)
#         padding_side = self.tokenizer.padding_side
#         if padding_side == "right":
#             for label_name in label_names:
#                 batch[label_name] = [
#                     list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels[label_name]
#                 ]
#         else:
#             for label_name in label_names:
#                 batch[label_name] = [
#                     [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels[label_name]
#                 ]
#
#         batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
#         return batch


@dataclass
class DataCollatorForAutoFormat:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: Optional[Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]] = None
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: int = 128
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    training: bool = True

    def __call__(self, features):
        """ Make Meta Schema Batch

        Args:
            features (Dict): [description]
                - sample_prompt: indicates sample_prompt example, need pop after call
                - spots (List[str]): List of spots in this sentence, need pop after call
                - asocs (List[str]): List of asocs in this sentence, need pop after call
                - input_ids
                - attention_mask
                - labels

        Returns:
        """

        # output_features = list()
        for feature in features:
            new_feature = dict()
            for k, v in feature.items():
                # if k in {"input_ids", "attention_mask", "token_type_ids"}:
                #     new_feature[k] = torch.FloatTensor([v])
                # elif k in {"au_flag", "link_flag"}:
                new_feature[k] = torch.LongTensor(v)
                # else:
                #     continue

        # output_features = self.tokenizer.pad(
        #     output_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt"
        # )

        return new_feature

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


@dataclass
class DataCollatorForAutoFormatV2:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: Optional[Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]] = None
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: int = 128
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    training: bool = True

    def __call__(self, features):
        """ Make Meta Schema Batch

        Args:
            features (Dict): [description]
                - sample_prompt: indicates sample_prompt example, need pop after call
                - spots (List[str]): List of spots in this sentence, need pop after call
                - asocs (List[str]): List of asocs in this sentence, need pop after call
                - input_ids
                - attention_mask
                - labels

        Returns:
        """

        feature = features[0]
        new_feature = dict()
        for k, v in feature.items():
            new_feature[k] = torch.LongTensor(v)

        return new_feature

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


@dataclass
class DataCollatorForTokenClassification(object):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"], dtype=torch.int64).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


@dataclass
class DatasetSamplerForAutoFormat:
    """
    Dataset Sampler for Auto Format
    """
    down_sample_length: int = 4
    up_sample_length: int = 64

    def __call__(self, dataset):
        """ Call

        Args:
            dataset (Dataset): Dataset

        Returns:
            List[Dict]: List of batch
        """
        dataset_less_than_down_sample_length = dataset.filter(
            lambda x: len(x["input_ids"]) <= self.down_sample_length,
            num_proc=16,
            desc="dataset less than down sample length"
        )
        dataset_more_than_up_sample_length = dataset.filter(
            lambda x: len(x["input_ids"]) >= self.up_sample_length,
            num_proc=16,
            desc="dataset more than up sample length"
        )
        normal_dataset = dataset.filter(
            lambda x: self.down_sample_length < len(x["input_ids"]) < self.up_sample_length,
            num_proc=16,
            desc="normal dataset"
        )
        dataset_less_than_down_sample_length = dataset_less_than_down_sample_length.shuffle().select(range(10000))
        dataset = concatenate_datasets([
            dataset_less_than_down_sample_length,
            normal_dataset,
            normal_dataset
        ] + [dataset_more_than_up_sample_length] * 10)
        return dataset


# @dataclass
# class DatasetSamplerForConcatenateShortIntoLong:
#     """
#     Dataset Sampler for Short into Long
#     """
#     down_sample_length: int = 4
#     up_sample_length: int = 64
#
#     def __call__(self, dataset):
#         """ Call
#
#         Args:
#             dataset (Dataset): Dataset
#
#         Returns:
#             List[Dict]: List of batch
#         """
#         dataset_less_than_down_sample_length = dataset.filter(
#             lambda x: len(x["input_ids"]) <= self.down_sample_length,
#             num_proc=16,
#             desc="dataset less than down sample length"
#         )
#         dataset_more_than_up_sample_length = dataset.filter(
#             lambda x: len(x["input_ids"]) >= self.up_sample_length,
#             num_proc=16,
#             desc="dataset more than up sample length"
#         )
#         normal_dataset = dataset.filter(
#             lambda x: self.down_sample_length < len(x["input_ids"]) < self.up_sample_length,
#             num_proc=16,
#             desc="normal dataset"
#         )
#         dataset_less_than_down_sample_length = dataset_less_than_down_sample_length.shuffle().select(range(10000))
#         dataset = concatenate_datasets([
#             dataset_less_than_down_sample_length,
#             normal_dataset,
#             normal_dataset
#         ] + [dataset_more_than_up_sample_length] * 10)
#         return dataset


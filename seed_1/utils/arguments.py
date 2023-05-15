import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(default="none", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


@dataclass
class TokenPairBaseDataTrainingArguments(BaseDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="none", metadata={"help": "The name of the task (ner, pos...)."})
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    stride_window: Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "The stride window for tokenization. "
            )
        },
    )
    is_split_into_words: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to add special tokens or not."
            )
        },
    )
    length_enhancement: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Whether to sample and concat from all samples to model maximum sentence length. "
                "If none zero, will sample from all samples and concat it to make the length exceed the length "
                "enhancement value."
            )
        }
    )


@dataclass
class TokenPairNERDataTrainingArguments(TokenPairBaseDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    entity_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    entity_types: str = field(
        default="", metadata={"help": "The file containing all entity type(a JSON file)."}
    )
    return_entity_level_metrics: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        super().__post_init__()
        extension = self.entity_types.split(".")[-1]
        assert extension in ["json"], "`entity_types` should be a json file."
        entity_types = json.load(open(self.entity_types, "r"))
        self.entity_types = entity_types
        self.entity2id = {entity_types[i]: i for i in range(len(entity_types))}
        self.id2entity = {i: entity_types[i] for i in range(len(entity_types))}
        self.label2id = self.entity2id
        self.id2label = self.id2entity


@dataclass
class TokenPairEREDataTrainingArguments(TokenPairNERDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ere", metadata={"help": "The name of the task (ner, pos...)."})
    relation_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of relation to input in the file (a csv or JSON file)."}
    )
    relation_types: Optional[str] = field(
        default=None, metadata={"help": "The file containing all relation type(a JSON file)."}
    )
    directed_relation: Optional[bool] = field(
        default=False, metadata={"help": "Whether the relation is directed or not."}
    )
    label_types: Optional[str] = field(
        default=None, metadata={"help": "The file containing all tag type(a JSON file)."}
    )
    return_relation_level_metrics: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return all the relation levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.relation_types is not None or self.label_types is None, "labels should be provided."

        if self.relation_types is not None:
            extension = self.relation_types.split(".")[-1]
            assert extension in ["json"], "`relation_types` should be a json file."
            relation_types = json.load(open(self.relation_types, "r"))
            self.relation_types = relation_types
            self.relation2id = {relation_types[i]: i for i in range(len(relation_types))}
            self.id2relation = {i: relation_types[i] for i in range(len(relation_types))}

        if self.label_types is not None:
            extension = self.label_types.split(".")[-1]
            assert extension in ["json"], "`tag_types` should be a json file."
            label_types = json.load(open(self.label_types, "r"))
            self.label_types = label_types
        elif self.directed_relation:
            relation_types = [f"{r}#{t}" for r in self.relation_types for t in ["SH2OH", "ST2OT", "OH2SH", "OT2ST"]]
            self.label_types = self.entity_types + relation_types
        else:
            relation_types = [f"{r}#{t}" for r in self.relation_types for t in ["H2H", "T2T"]]
            self.label_types = self.entity_types + relation_types
        self.label2id = {self.label_types[i]: i for i in range(len(self.label_types))}
        self.id2label = {i: self.label_types[i] for i in range(len(self.label_types))}


@dataclass
class TokenClassificationDataTrainingArguments(BaseDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."},)
    entity_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."},
    )
    entity_types: Optional[str] = field(
        default=None, metadata={"help": "The file containing all entity type(a JSON file)."},
    )
    tag_types: Optional[str] = field(
        default=None, metadata={"help": "The file containing all tag type(a JSON file)."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    stride_window: int = field(
        default=128,
        metadata={
            "help": (
                "The stride window to use when tokenizing. If set, the tokens will be split in overlapping windows "
            )
        },
    )
    is_split_into_words: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    add_special_tokens: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: Optional[bool]= field(
        default=True,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    span_enhancement: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use span enhancement for training."},
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.entity_types is not None or self.tag_types is not None, \
            "You must specify `entity_types` or `tag_types`."
        if self.entity_types is not None:
            extension = self.entity_types.split(".")[-1]
            assert extension in ["json"], "`entity_types` should be a json file."
            entity_types = json.load(open(self.entity_types, "r"))
            self.entity_types = entity_types
            self.entity2id = {entity_types[i]: i for i in range(len(entity_types))}
            self.id2entity = {i: entity_types[i] for i in range(len(entity_types))}
        else:
            extension = self.tag_types.split(".")[-1]
            assert extension in ["json"], "`tag_types` should be a json file."
            tag_types = json.load(open(self.tag_types, "r"))
            entity_types = [tag_type.split("-")[-1] for tag_type in tag_types]
            self.entity_types = entity_types
            self.entity2id = {entity_types[i]: i for i in range(len(entity_types))}
            self.id2entity = {i: entity_types[i] for i in range(len(entity_types))}

        if self.tag_types is not None:
            extension = self.tag_types.split(".")[-1]
            assert extension in ["json"], "`tag_types` should be a json file."
            tag_types = json.load(open(self.tag_types, "r"))
            self.tag2id = {tag_types[i]: i for i in range(len(tag_types))}
            self.id2tag = {i: tag_types[i] for i in range(len(tag_types))}
        else:
            tag_types = (["O"] if "O" in self.entity_types else []) + [
                f"{prefix}-{entity_type}"
                for entity_type in self.entity_types for prefix in ["B", "I"]
                if entity_type != "O"
            ]
            self.tag_types = tag_types
            self.tag2id = {tag_types[i]: i for i in range(len(tag_types))}
            self.id2tag = {i: tag_types[i] for i in range(len(tag_types))}
        # if self.span_enhancement:
        #     self.span_list = None
        #     self.span_label2id = None
        #     self.id2span_label = None


@dataclass
class AutoFormatDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="Auto Format", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    entity_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of entity to input in the file (a csv or JSON file)."}
    )
    relation_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of relation to input in the file (a csv or JSON file)."}
    )
    label_list_file: Optional[str] = field(
        default=None, metadata={"help": "The file containing all label type(a JSON file)."}
    )
    entity_label_file: Optional[str] = field(
        default=None, metadata={"help": "The file containing all entity type(a JSON file)."}
    )
    relation_label_file: Optional[str] = field(
        default=None, metadata={"help": "The file containing all relation type(a JSON file)."}
    )
    directed: bool = field(
        default=False, metadata={"help": "Whether the relation is directed or not."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    model_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    add_prefix_space: Optional[bool] = field(
        default=False,
        metadata={"help": "Add a space before each token for pretokenized datasets."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    pad_to_max_length: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    is_split_into_words: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_details: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    kernel_size: Optional[int] = field(
        default=128,
        metadata={"help": "The kernel size for the convolutional layer."},
    )
    hidden_size: Optional[int] = field(
        default=384,
        metadata={"help": "The hidden size for the convolutional layer."},
    )
    num_layers: Optional[int] = field(
        default=2,
        metadata={"help": "The number of layers for the convolutional layer."},
    )
    num_heads: Optional[int] = field(
        default=8,
        metadata={"help": "The number of heads for the attention layer."},
    )
    attention_size: Optional[int] = field(
        default=32,
        metadata={"help": "The head size for the attention layer."},
    )
    attention_type: Optional[str] = field(
        default='non_symmetric',
        metadata={"help": "The attention type for the attention layer."},
    )
    value_size: Optional[int] = field(
        default=32,
        metadata={"help": "The value size for the attention layer."},
    )
    node_output_size: Optional[int] = field(
        default=2,
        metadata={"help": "The output size for the node layer."},
    )
    link_output_size: Optional[int] = field(
        default=1,
        metadata={"help": "The output size for the link layer."},
    )
    scale: Optional[int] = field(
        default=8,
        metadata={"help": "The scale for the link layer."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()
        self.label2id = None
        self.id2label = None


@dataclass
class AutoFormatV2DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="Auto Format V2", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    entity_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of entity to input in the file (a csv or JSON file)."}
    )
    relation_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of relation to input in the file (a csv or JSON file)."}
    )
    entity_label_file: Optional[str] = field(
        default=None, metadata={"help": "The file containing all entity type(a JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    model_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    ),
    add_prefix_space: Optional[bool] = field(
        default=False,
        metadata={"help": "Add a space before each token for pretokenized datasets."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    is_split_into_words: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    return_details: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    kernel_size: Optional[int] = field(
        default=128,
        metadata={"help": "The kernel size for the convolutional layer."},
    )
    hidden_size: Optional[int] = field(
        default=384,
        metadata={"help": "The hidden size for the convolutional layer."},
    )
    num_layers: Optional[int] = field(
        default=2,
        metadata={"help": "The number of layers for the convolutional layer."},
    )
    num_heads: Optional[int] = field(
        default=8,
        metadata={"help": "The number of heads for the attention layer."},
    )
    attention_size: Optional[int] = field(
        default=32,
        metadata={"help": "The head size for the attention layer."},
    )
    attention_type: Optional[str] = field(
        default='non_symmetric',
        metadata={"help": "The attention type for the attention layer."},
    )
    value_size: Optional[int] = field(
        default=32,
        metadata={"help": "The value size for the attention layer."},
    )
    node_output_size: Optional[int] = field(
        default=2,
        metadata={"help": "The output size for the node layer."},
    )
    link_output_size: Optional[int] = field(
        default=1,
        metadata={"help": "The output size for the link layer."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()
        self.label2id = None
        self.id2label = None


@dataclass
class ReplugDataTrainingArguments(BaseDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default="text similarity",
        metadata={"help": "The name of the task (ner, pos...)."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )


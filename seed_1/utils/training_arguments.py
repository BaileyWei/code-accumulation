from dataclasses import dataclass, field
from transformers import TrainingArguments


# @dataclass
# class PLTrainingArguments(TrainingArguments):


@dataclass
class TokenPairNERTrainingArguments(TrainingArguments):
    """
    Parameters:
        tag2idx (:obj:`dict`, `optional`, defaults to :obj:`None`):
        idx2tag (:obj:`dict`, `optional`, defaults to :obj:`None`):
            Whether to use Constraint Decoding
        evaluate_by_labels (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    tag2idx: dict = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    idx2tag: dict = field(default=False, metadata={"help": "Whether to save better metric checkpoint"})
    evaluate_by_labels: bool = field(default=False, metadata={"help": "whether to evaluate by label"})
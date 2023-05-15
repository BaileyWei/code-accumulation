from typing import Dict, List, Union
import math
import torch
import numpy as np
import pytorch_lightning as pl
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)
from transformers.trainer_pt_utils import nested_numpify

# todo: from seed.utils.training_arguments import PLTrainingArguments
from seed.utils.training_arguments import TokenPairNERTrainingArguments


class BasePLModule(pl.LightningModule):
    def __init__(
            self,
            training_args: TrainingArguments,
            tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
            model: Union[PreTrainedModel, torch.nn.Module],
    ):
        super().__init__()

        self.training_args = training_args
        self.tokenizer = tokenizer
        self.model = model

        self.prepare_training_args()

    def forward(self, inputs):

        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)

        return outputs

    def training_step(self, batch: Dict, batch_idx: int):

        outputs = self.model(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        # self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict[str, float]:

        labels = nested_numpify(batch.pop("labels"))
        outputs = self.model(**batch)
        predictions = nested_numpify((outputs[0] > 0).long())

        return self.compute_metrics(predictions, labels)

    def validation_epoch_end(self, outputs: List[Dict]) -> None:

        metrics = dict()
        for output in outputs:
            for key, value in output.items():
                metrics[key] = metrics.get(key, 0) + value
        metrics = self.gather_metrics(metrics)
        self.log_dict(metrics, prog_bar=True, logger=True)

    def compute_metrics(self, predictions: np.array, labels: np.array) -> Dict[str, float]:
        raise NotImplementedError

    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError

    def configure_optimizers(self):
        """
        Prepare optimizer and schedule (linear warmup and decay)
        reproduce the same results as the transformers library
        since transformers update scheduler after each step,
        we will not return scheduler in this function
        return: optimizer, scheduler
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.training_args.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # todo: or "epoch", intagrate it into training arguments
                "frequency": 1,  # todo: intagrate it into training arguments
            },
        }

    def prepare_training_args(self) -> None:
        """
        Prepare training arguments
        """
        if not self.training_args.max_steps:

            self.training_args.max_steps = math.ceil(
                self.training_args.num_train_epochs
                * len(self.train_dataloader())
                // self.training_args.gradient_accumulation_steps
            )
        self.log("max_steps", self.training_args.max_steps, logger=True)

class TokenPairNERPLModule(BasePLModule):
    def __init__(
            self,
            training_args: TokenPairNERTrainingArguments,
            tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
            model: Union[PreTrainedModel, torch.nn.Module],
    ):
        super().__init__(training_args, tokenizer, model)

        # self.tag2idx = tag2idx
        # self.idx2tag = {v: k for k, v in tag2idx.items()}

    def forward(self, inputs):
        """Todo: implement subword mask"""

        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        outputs = (outputs.logits > 0).long().nonzero(as_tuple=True)

        return outputs

    def compute_metrics(self, predictions: np.array, labels: np.array) -> Dict[str, float]:
        # todo: add support for multi-label
        result = dict()
        predictions = np.where(labels == -100, 0, predictions)
        labels = np.where(labels == -100, 0, labels)

        result["tp"], result["fp"], result["fn"] = 0, 0, 0
        for prediction, label in zip(predictions, labels):
            pred_pos, pred_typ = np.nonzero(prediction)
            pred = [(p, t) for p, t in zip(pred_pos.tolist(), pred_typ.tolist())]
            gold_pos, gold_typ = np.nonzero(label)
            gold = [(p, t) for p, t in zip(gold_pos.tolist(), gold_typ.tolist())]
            pred, gold = set(pred), set(gold)
            result["tp"] += len(pred & gold)
            result["fp"] += len(pred - gold)
            result["fn"] += len(gold - pred)
            # if self.training_args.evaluate_by_labels:
            #     for
            #     result[f"{self.training_args.idx2tag[]}_tp"] += len(pred & gold)

        return result

    def gather_metrics(self, metrics):

        epsilon = 1e-12
        tp = metrics.pop("tp", 0)
        fp = metrics.pop("fp", 0)
        fn = metrics.pop("fn", 0)
        metrics["precision"] = round(tp / (tp + fp + epsilon), 6)
        metrics["recall"] = round(tp / (tp + fn + epsilon), 6)
        metrics['f1-score'] = round(2 * (metrics["precision"] * metrics["recall"]) / \
                                    (metrics["precision"] + metrics["recall"] + epsilon), 6)

        return metrics



# todo: implement token pair relation extraction module
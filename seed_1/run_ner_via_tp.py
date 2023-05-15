#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os, sys
import json
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import datasets
import numpy as np
from datasets import ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from seed.models.token_pair_models import RobertaForTokenPairClassification
from seed.utils.trainers import TrainerForLargeValidationDataset
from seed.utils.data_collectors import DataCollatorForTokenPairNER
from seed.utils.arguments import TokenPairNERDataTrainingArguments
from seed.utils.tools import TokenPairNERDataOperator

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    tp_kernel: str = field(
        default="EGP",
        metadata={"help": "Which kernel to use for token pair classification"},
    )
    head_size: Optional[int] = field(
        default=64,
        metadata={"help": "head size of the token pair classifier"},
    )
    rope: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to use rope in tp kernel"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, TokenPairNERDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    # def get_label_list(labels):
    #     unique_labels = set()
    #     for label in labels:
    #         unique_labels = unique_labels | set(label)
    #     label_list = list(unique_labels)
    #     label_list.sort()
    #     return label_list
    label_list = json.load(open(data_args.label_list_file))
    num_labels = len(label_list)

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    # labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    # if labels_are_int:
    #     label_list = features[label_column_name].feature.names
    #     label_to_id = {i: i for i in range(len(label_list))}
    # else:
    #     label_list = get_label_list(raw_datasets["train"][label_column_name])
    #     label_to_id = {l: i for i, l in enumerate(label_list)}
    label_to_id = {l: i for i, l in enumerate(label_list)}

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    task_specific_params = {
        "tp_kernel": model_args.tp_kernel,
        "head_size": model_args.head_size,
        "output_size": num_labels,
        "rope": True
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        task_specific_params=task_specific_params
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = RobertaForTokenPairClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Model has labels -> use them.
    # if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
    #     if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
    #         # Reorganize `label_list` to match the ordering of the model.
    #         # if labels_are_int:
    #         #     label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
    #         #     label_list = [model.config.id2label[i] for i in range(num_labels)]
    #         # else:
    #         label_list = [model.config.id2label[i] for i in range(num_labels)]
    #         label_to_id = {l: i for i, l in enumerate(label_list)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
    #             f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
    #         )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    # b_to_i_label = []
    # for idx, label in enumerate(label_list):
    #     if label.startswith("B-") and label.replace("B-", "I-") in label_list:
    #         b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    #     else:
    #         b_to_i_label.append(idx)

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    data_operator = TokenPairNERDataOperator(data_args=data_args, tokenizer=tokenizer)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = data_operator.operate(train_dataset)
            # train_dataset = train_dataset.map(
            #     tokenize_and_align_labels,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Running tokenizer on train dataset",
            # )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = data_operator.operate(eval_dataset)
            # eval_dataset = eval_dataset.map(
            #     tokenize_and_align_labels,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Running tokenizer on validation dataset",
            # )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = data_operator.operate(predict_dataset)
            # predict_dataset = predict_dataset.map(
            #     tokenize_and_align_labels,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Running tokenizer on prediction dataset",
            # )

    # Data collator
    data_collator = DataCollatorForTokenPairNER(tokenizer, num_of_tags=len(label_list))

    # Metrics
    def compute_metrics(eval_preds):

        logits, labels = eval_preds.predictions, eval_preds.label_ids
        predictions = (logits > 0).astype(int)

        true_predictions = np.where(labels == -100, 0, predictions)
        true_labels = np.where(labels == -100, 0, labels)

        result = defaultdict(int)
        for prediction, label in zip(true_predictions, true_labels):
            pred_pos, pred_typ = np.nonzero(prediction)
            pred = [(i, j) for i, j in zip(pred_pos.tolist(), pred_typ.tolist())]
            gold_pos, gold_typ = np.nonzero(label)
            gold = [(i, j) for i, j in zip(gold_pos.tolist(), gold_typ.tolist())]
            pred, gold = set(pred), set(gold)
            result["tp"] += len(pred & gold)
            result["fp"] += len(pred - gold)
            result["fn"] += len(gold - pred)
            if data_args.return_entity_level_metrics:
                # Unpack nested dictionaries
                pred_detail, gold_detail = defaultdict(set), defaultdict(set)
                for p, i in pred:
                    pred_detail[i].add(p)
                for p, i in gold:
                    gold_detail[i].add(p)
                for i in model.config.id2label:
                    result[f"{label_list[i]}_tp"] += len(pred_detail[i] & gold_detail[i])
                    result[f"{label_list[i]}_fp"] += len(pred_detail[i] - gold_detail[i])
                    result[f"{label_list[i]}_fn"] += len(gold_detail[i] - pred_detail[i])

        return result

    def gather_metrics(metrics):

        epsilon = 1e-12
        tp = metrics.pop("tp", 0)
        fp = metrics.pop("fp", 0)
        fn = metrics.pop("fn", 0)
        metrics["precision"] = round(tp / (tp + fp + epsilon), 6)
        metrics["recall"] = round(tp / (tp + fn + epsilon), 6)
        metrics['f1-score'] = round(2 * (metrics["precision"] * metrics["recall"]) / \
                                    (metrics["precision"] + metrics["recall"] + epsilon), 6)
        if data_args.return_entity_level_metrics:
            for i in model.config.id2label:
                tp = metrics.pop(f"{label_list[i]}_tp", 0)
                fp = metrics.pop(f"{label_list[i]}_fp", 0)
                fn = metrics.pop(f"{label_list[i]}_fn", 0)
                metrics[f"{label_list[i]}_precision"] = round(tp / (tp + fp + epsilon), 6)
                metrics[f"{label_list[i]}_recall"] = round(tp / (tp + fn + epsilon), 6)
                metrics[f"{label_list[i]}_f1-score"] = round(2 * (metrics[f"{label_list[i]}_precision"] * \
                                                                  metrics[f"{label_list[i]}_recall"]) / \
                                                            (metrics[f"{label_list[i]}_precision"] + \
                                                             metrics[f"{label_list[i]}_recall"] + epsilon), 6)

        return metrics

    # Initialize our Trainer
    trainer = TrainerForLargeValidationDataset(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        gather_metrics=gather_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
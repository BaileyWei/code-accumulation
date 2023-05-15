import logging
import copy
from typing import Dict, Union, Any, Optional, List

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import datasets
from datasets import Dataset
from transformers import Trainer
from transformers.utils import is_datasets_available, is_torch_tpu_available, logging
from transformers.deepspeed import deepspeed_init
from transformers.trainer_utils import seed_worker, has_length, denumpify_detensorize, EvalLoopOutput, EvalPrediction
from transformers.trainer_pt_utils import IterableDatasetShard, find_batch_size, \
    nested_concat, nested_numpify, nested_truncate


logger = logging.get_logger(__name__)


class DebuggingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        # train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            # sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        error_flag = False
        try:
            loss = super().training_step(model, inputs)
            return loss
        except Exception as e:
            error_flag = True
            error = e
        if error_flag:
            print("Error Occur when training with following data")
            print(f"input data: {inputs}")
            # print(f"input shape: {inputs['input_ids'].size()}")
            # for input_data in inputs['input_ids']:
            #     print(input_data)
            raise error


class TrainerForLargeValidationDataset(Trainer):

    def __init__(self, gather_metrics=None, dataset_sampler=None, **kwargs):
        super().__init__(**kwargs)
        self.gather_metrics = gather_metrics
        self.dataset_sampler = dataset_sampler

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_metrics = dict()
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            loss, logits, labels = nested_numpify(loss), nested_numpify(logits), nested_numpify(labels)

            if self.compute_metrics is not None and logits is not None and labels is not None:
                if args.include_inputs_for_metrics:
                    step_metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, inputs=all_inputs)
                    )
                else:
                    step_metrics = self.compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))
            else:
                step_metrics = {}

            for k, v in step_metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = 0
                all_metrics[k] += step_metrics[k]
            # inputs_decode = inputs["input_ids"] if args.include_inputs_for_metrics else None

            # if is_torch_tpu_available():
            #     xm.mark_step()

            # Update containers on host
            # if loss is not None:
            #     losses = self._nested_gather(loss.repeat(batch_size))
            #     losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            # if labels is not None:
            #     labels = self._pad_across_processes(labels)
            #     labels = self._nested_gather(labels)
            #     labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            # if inputs_decode is not None:
            #     inputs_decode = self._pad_across_processes(inputs_decode)
            #     inputs_decode = self._nested_gather(inputs_decode)
            #     inputs_host = (
            #         inputs_decode
            #         if inputs_host is None
            #         else nested_concat(inputs_host, inputs_decode, padding_index=-100)
            #     )
            # if logits is not None:
            #     logits = self._pad_across_processes(logits)
            #     logits = self._nested_gather(logits)
            #     if self.preprocess_logits_for_metrics is not None:
            #         logits = self.preprocess_logits_for_metrics(logits, labels)
            #     preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            # if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
            #     if losses_host is not None:
            #         losses = nested_numpify(losses_host)
            #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
            #     if preds_host is not None:
            #         logits = nested_numpify(preds_host)
            #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            #     if inputs_host is not None:
            #         inputs_decode = nested_numpify(inputs_host)
            #         all_inputs = (
            #             inputs_decode
            #             if all_inputs is None
            #             else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            #         )
            #     if labels_host is not None:
            #         labels = nested_numpify(labels_host)
            #         all_labels = (
            #             labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
            #         )
            #
            #     # Set back to None to begin a new accumulation
            #     losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        # if losses_host is not None:
        #     losses = nested_numpify(losses_host)
        #     all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        # if preds_host is not None:
        #     logits = nested_numpify(preds_host)
        #     all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        # if inputs_host is not None:
        #     inputs_decode = nested_numpify(inputs_host)
        #     all_inputs = (
        #         inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
        #     )
        # if labels_host is not None:
        #     labels = nested_numpify(labels_host)
        #     all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        # if all_losses is not None:
        #     all_losses = all_losses[:num_samples]
        # if all_preds is not None:
        #     all_preds = nested_truncate(all_preds, num_samples)
        # if all_labels is not None:
        #     all_labels = nested_truncate(all_labels, num_samples)
        # if all_inputs is not None:
        #     all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.gather_metrics is not None and all_metrics is not None:
            metrics = self.gather_metrics(all_metrics)

        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if self.dataset_sampler:
            train_dataset = self.dataset_sampler(train_dataset)
        data_collator = copy.copy(self.data_collator)
        if hasattr(data_collator, "train"):
            data_collator.train()
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = copy.copy(self.data_collator)
        if hasattr(data_collator, "eval"):
            data_collator.eval()
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
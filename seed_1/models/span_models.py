from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
    BertModel,
    BertPreTrainedModel,
    DebertaModel,
    DebertaPreTrainedModel,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    DistilBertModel,
    DistilBertPreTrainedModel,
    ElectraModel,
    ElectraPreTrainedModel,
    RobertaModel,
    RobertaPreTrainedModel
)
from transformers.modeling_outputs import TokenClassifierOutput

from seed.utils.outputs import SpanOutput
from seed.components.span_prediction import PoolingLogits
from seed.layers.losses import (
    BCEWithLogitsLossWithMask,
    LabelSmoothingCrossEntropy, 
    FocalLoss,
    MultiLabelCategoricalCrossEntropy
)


@dataclass
class SpanModelArguments:
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

    soft_label: bool = field(
        default=False,
        metadata={"help": "Whether to use soft label for training."},
    )
    loss_type: str = field(
        default="BCE",
        metadata={"help": "Loss type, one of ['CE', 'BCE', 'LSCE', 'FL', 'MLCE']"},
    )
    model_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    ),
    add_prefix_space: Optional[bool] = field(
        default=False,
        metadata={"help": "Add a space before each token for pretokenized datasets."},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


class AlbertForTokenPairClassification(AlbertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        config.update(config.task_specific_params)
        self.config = config

        # setattr(self, config.model_type, AutoModel(config))
        self.albert = AlbertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
        if config.soft_label:
            self.end_fc = nn.Sequential(
                PoolingLogits(
                    config.hidden_size + config.num_labels,
                    config.layer_norm if hasattr(config, 'layer_norm') else False
                ),
                nn.Linear(config.hidden_size + config.num_labels, config.num_labels)
            )
        else:
            self.config.soft_label = False
            self.end_fc = nn.Linear(config.hidden_size, config.num_labels)

        assert config.loss_type in {"BCE", "MLCE"}
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_labels: Optional[torch.LongTensor] = None,
        end_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpanOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if self.config.soft_label:
            if start_labels is None:
                start_labels = (start_logits > 0).long()
            end_logits = self.end_fc(torch.cat([sequence_output, start_labels.float()], dim=-1))
        else:
            end_logits = self.end_fc(sequence_output)

        loss = None
        if start_labels is not None and end_labels is not None:
            if self.config.loss_type == "BCE":
                loss_fct = BCEWithLogitsLossWithMask()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            else:
                loss_fct = MultiLabelCategoricalCrossEntropy()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpanOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForSpanPrediction(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        config.update(config.task_specific_params)
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
        if config.soft_label:  # and config.loss_type in {"BCE", "MLCE"}:# soft label will not work with CE, Focal, LSR
            self.end_fc = nn.Sequential(
                PoolingLogits(
                    config.hidden_size + config.num_labels,
                    config.layer_norm if hasattr(config, 'layer_norm') else False
                ),
                nn.Linear(config.hidden_size + config.num_labels, config.num_labels)
            )
            # if config.loss_type in {"CE", "LSCE", "FL"}:
            #     self.label_embedding = nn.Embedding(config.num_labels, config.num_labels)
        else:
            config.soft_label = False
            self.end_fc = nn.Linear(config.hidden_size, config.num_labels)

        assert config.loss_type in {"BCE", "MLCE"}  # {"CE", "BCE", "LSCE", "FL", "MLCE"}
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_labels: Optional[torch.LongTensor] = None,
        end_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpanOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if self.config.soft_label:
            # if start_labels is not None:
            #     # if self.config.loss_type in {"CE", "LSCE", "FL"}:
            #     #     start_logits = self.label_embedding(start_logits).squeeze(-2)
            #     end_logits = self.end_fc(torch.cat([sequence_output, start_labels.float()], dim=-1))
            # else:
            #     start_labels = (start_logits > 0).float()
            #     end_logits = self.end_fc(torch.cat([sequence_output, start_labels], dim=-1))
            if start_labels is None:
                start_labels = (start_logits > 0).long()
            end_logits = self.end_fc(torch.cat([sequence_output, start_labels.float()], dim=-1))
        else:
            end_logits = self.end_fc(sequence_output)

        loss = None
        if start_labels is not None and end_labels is not None:
            # if self.config.loss_type == "CE":
            #     loss_fct = CrossEntropyLoss()
            #     start_loss = loss_fct(start_logits.view(-1, self.config.num_labels), start_labels.view(-1))
            #     end_loss = loss_fct(end_logits.view(-1, self.config.num_labels), end_labels.view(-1))
            # elif self.config.loss_type == "BCE":
            if self.config.loss_type == "BCE":
                loss_fct = BCEWithLogitsLossWithMask()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            # elif self.config.loss_type == "LSCE":
            #     # label should be like [1, 0, 3, 5, 0, 1]
            #     loss_fct = LabelSmoothingCrossEntropy()
            #     start_loss = loss_fct(start_logits.view(-1, self.config.num_labels), start_labels.view(-1))
            #     end_loss = loss_fct(end_logits.view(-1, self.config.num_labels), end_labels.view(-1))
            # elif self.config.loss_type == "FL":
            #     # label should be like [1, 0, 3, 5, 0, 1]
            #     loss_fct = FocalLoss()
            #     start_loss = loss_fct(
            #         start_logits.view(-1, self.config.num_labels), start_labels.view(-1)
            #     )
            #     end_loss = loss_fct(
            #         end_logits.view(-1, self.config.num_labels), end_labels.view(-1)
            #     )
            else:  # self.config.loss_type == "MLCE"
                loss_fct = MultiLabelCategoricalCrossEntropy()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpanOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaForTokenPairClassification(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.task_specific_params:
            config.update(config.task_specific_params)
            config.fusion_num = len(config.fusion_layer)
        self.config = config

        self.deberta = DebertaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
        if config.soft_label:
            self.end_fc = nn.Sequential(
                PoolingLogits(
                    config.hidden_size + config.num_labels,
                    config.layer_norm if hasattr(config, 'layer_norm') else False
                ),
                nn.Linear(config.hidden_size + config.num_labels, config.num_labels)
            )
        else:
            self.config.soft_label = False
            self.end_fc = nn.Linear(config.hidden_size, config.num_labels)

        assert config.loss_type in {"BCE", "MLCE"}
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_labels: Optional[torch.LongTensor] = None,
        end_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpanOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if self.config.soft_label:
            if start_labels is None:
                start_labels = (start_logits > 0).long()
            end_logits = self.end_fc(torch.cat([sequence_output, start_labels.float()], dim=-1))
        else:
            end_logits = self.end_fc(sequence_output)

        loss = None
        if start_labels is not None and end_labels is not None:
            if self.config.loss_type == "BCE":
                loss_fct = BCEWithLogitsLossWithMask()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            else:
                loss_fct = MultiLabelCategoricalCrossEntropy()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpanOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV2ForTokenPairClassification(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        config.update(config.task_specific_params)
        self.config = config

        self.deberta = DebertaV2Model(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
        if config.soft_label:
            self.end_fc = nn.Sequential(
                PoolingLogits(
                    config.hidden_size + config.num_labels,
                    config.layer_norm if hasattr(config, 'layer_norm') else False
                ),
                nn.Linear(config.hidden_size + config.num_labels, config.num_labels)
            )
        else:
            self.config.soft_label = False
            self.end_fc = nn.Linear(config.hidden_size, config.num_labels)

        assert config.loss_type in {"BCE", "MLCE"}
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_labels: Optional[torch.LongTensor] = None,
        end_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpanOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if self.config.soft_label:
            if start_labels is None:
                start_labels = (start_logits > 0).long()
            end_logits = self.end_fc(torch.cat([sequence_output, start_labels.float()], dim=-1))
        else:
            end_logits = self.end_fc(sequence_output)

        loss = None
        if start_labels is not None and end_labels is not None:
            if self.config.loss_type == "BCE":
                loss_fct = BCEWithLogitsLossWithMask()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            else:
                loss_fct = MultiLabelCategoricalCrossEntropy()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpanOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DistilBertForTokenPairClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.update(config.task_specific_params)
        self.config = config

        self.distilbert = DistilBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
        if config.soft_label:
            self.end_fc = nn.Sequential(
                PoolingLogits(
                    config.hidden_size + config.num_labels,
                    config.layer_norm if hasattr(config, 'layer_norm') else False
                ),
                nn.Linear(config.hidden_size + config.num_labels, config.num_labels)
            )
        else:
            self.config.soft_label = False
            self.end_fc = nn.Linear(config.hidden_size, config.num_labels)

        assert config.loss_type in {"BCE", "MLCE"}
        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_labels: Optional[torch.LongTensor] = None,
        end_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpanOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if self.config.soft_label:
            if start_labels is None:
                start_labels = (start_logits > 0).long()
            end_logits = self.end_fc(torch.cat([sequence_output, start_labels.float()], dim=-1))
        else:
            end_logits = self.end_fc(sequence_output)

        loss = None
        if start_labels is not None and end_labels is not None:
            if self.config.loss_type == "BCE":
                loss_fct = BCEWithLogitsLossWithMask()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            else:
                loss_fct = MultiLabelCategoricalCrossEntropy()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpanOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ElectraForTokenPairClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.update(config.task_specific_params)
        self.config = config

        self.electra = ElectraModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
        if config.soft_label:
            self.end_fc = nn.Sequential(
                PoolingLogits(
                    config.hidden_size + config.num_labels,
                    config.layer_norm if hasattr(config, 'layer_norm') else False
                ),
                nn.Linear(config.hidden_size + config.num_labels, config.num_labels)
            )
        else:
            self.config.soft_label = False
            self.end_fc = nn.Linear(config.hidden_size, config.num_labels)

        assert config.loss_type in {"BCE", "MLCE"}
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_labels: Optional[torch.LongTensor] = None,
        end_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpanOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states  = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        start_logits = self.start_fc(discriminator_sequence_output)
        if self.config.soft_label:
            if start_labels is None:
                start_labels = (start_logits > 0).long()
            end_logits = self.end_fc(torch.cat([discriminator_sequence_output, start_labels.float()], dim=-1))
        else:
            end_logits = self.end_fc(discriminator_sequence_output)

        loss = None
        if start_labels is not None and end_labels is not None:
            if self.config.loss_type == "BCE":
                loss_fct = BCEWithLogitsLossWithMask()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            else:
                loss_fct = MultiLabelCategoricalCrossEntropy()
                start_loss = loss_fct(
                    start_logits.view(-1, self.config.num_labels), start_labels.float().view(-1, self.config.num_labels)
                )
                end_loss = loss_fct(
                    end_logits.view(-1, self.config.num_labels), end_labels.float().view(-1, self.config.num_labels)
                )
            loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits,) + discriminator_hidden_states[2:]
            return ((loss,) + output) if loss is not None else output

        return SpanOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


# class RobertaForFusionTokenClassification(RobertaPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids"]
#
#     def __init__(self, config):
#         super().__init__(config)
#
#         if config.task_specific_params:
#             config.update(config.task_specific_params)
#             config.fusion_num = len(config.fusion_layer)
#         self.config = config
#
#         self.roberta = RobertaModel(config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(config.hidden_size * (config.fusion_num + 1), config.num_labels)
#
#         # self.span_classifier = nn.Linear(config.hidden_size * (config.fusion_num + 1), 2 * len(config.label_list))
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         # input_features: Optional[torch.FloatTensor] = None,
#         span_labels: Optional[torch.FloatTensor] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], TokenClassifierV2Output]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=self.config.fusion,  # output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         intermediate_output = list()
#         if self.config.fusion:
#             for fusion_layer in self.config.fusion_layer:
#                 intermediate_output.append(outputs.hidden_states[fusion_layer])
#             sequence_output = [sequence_output] + intermediate_output
#             sequence_output = torch.cat(sequence_output, dim=-1)
#
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#         # span_logits = self.span_classifier(sequence_output)
#
#         loss = 0.0
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss += loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#
#         # if span_labels is not None:
#         #     span_loss_fct = CrossEntropyLoss()
#         #     loss += span_loss_fct(span_logits.view(-1, 2 * len(self.config.label_list)), span_labels.view(-1))
#
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#
#         return TokenClassifierV2Output(
#             loss=loss,
#             logits=logits,
#             # span_logits=span_logits,
#             hidden_states=None,  # outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

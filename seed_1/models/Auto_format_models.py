from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertPreTrainedModel,
)
from torch.nn import CrossEntropyLoss

from seed.components.auto_format import AutoFormatComponent, AutoFormatV2Component, AutoFormatV3Component
from seed.layers.losses import MultiLabelCategoricalCrossEntropy
from seed.utils.outputs import AutoFormatOutput, AutoFormatV2Output


class BertForAutoFormat(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        config.update(config.task_specific_params)
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.auto_format = AutoFormatComponent(config)

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
        node_labels: Optional[torch.LongTensor] = None,
        link_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], AutoFormatOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # for input in [input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds]:
        #     if input is not None:
        #         input.squeeze_(0)

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
        if hasattr(outputs, "pooler_output"):
            sequence_output = outputs.pooler_output
        else:
            sequence_output = self.mean_pooling(outputs[0], attention_mask)

        sequence_output = sequence_output.unsqueeze(0)
        node_logits, link_logits = self.auto_format(sequence_output)

        loss = 0
        if node_labels is not None:
            node_loss_fct = CrossEntropyLoss()
            loss += node_loss_fct(node_logits.view(-1, 2), node_labels.view(-1))

        if link_labels is not None:
            link_loss_fct = MultiLabelCategoricalCrossEntropy()
            loss += link_loss_fct(link_logits.squeeze(), link_labels)

        if not return_dict:
            output = (node_logits, link_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return AutoFormatOutput(
            loss=loss,
            node_logits=node_logits,
            link_logits=link_logits,
        )

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BertForAutoFormatV2(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        config.update(config.task_specific_params)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.auto_format = AutoFormatV2Component(config)

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
        node_labels: Optional[torch.LongTensor] = None,
        au_labels: Optional[torch.LongTensor] = None,
        link_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], AutoFormatV2Output]:
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
            output_hidden_states=True,  # output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.mean_pooling(outputs.hidden_states, attention_mask)

        sequence_output = sequence_output.unsqueeze(0)
        au_logits, node_logits, link_logits = self.auto_format(sequence_output)

        loss = 0.0
        if node_labels is not None:
            node_loss_fct = CrossEntropyLoss()
            loss += node_loss_fct(node_logits.view(-1, self.num_labels), node_labels.view(-1))

        if au_labels is not None:
            link_loss_fct = MultiLabelCategoricalCrossEntropy()
            loss += link_loss_fct(au_logits.squeeze(), au_labels)

        if link_labels is not None:
            link_loss_fct = CrossEntropyLoss()
            loss += link_loss_fct(link_logits.squeeze(), link_labels)

        if not return_dict:
            output = (node_logits, au_logits, link_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return AutoFormatV2Output(
            loss=loss,
            node_logits=node_logits,
            au_logits=au_logits,
            link_logits=link_logits,
        )

    @staticmethod
    def mean_pooling(model_output: torch.FloatTensor, attention_mask: torch.FloatTensor) -> torch.FloatTensor:
        first_embeddings, last_embeddings = model_output[1], model_output[-1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(first_embeddings.size()).float()
        first_embeddings = torch.sum(first_embeddings * input_mask_expanded, 1)
        last_embeddings = torch.sum(last_embeddings * input_mask_expanded, 1)
        return (first_embeddings + last_embeddings) / (torch.clamp(input_mask_expanded.sum(1), min=1e-9) * 2)


# class BertForAutoFormatV3(BertPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#
#     def __init__(self, config):
#         super().__init__(config)
#
#         config.update(config.task_specific_params)
#         self.config = config
#         self.num_labels = config.num_labels
#
#         self.bert = BertModel(config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.auto_format = AutoFormatV3Component(config)
#
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
#         node_labels: Optional[torch.LongTensor] = None,
#         au_labels: Optional[torch.LongTensor] = None,
#         link_labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], AutoFormatV2Output]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=True,  # output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = self.mean_pooling(outputs.hidden_states, attention_mask)
#
#         sequence_output = sequence_output.unsqueeze(0)
#         au_logits, node_logits, link_logits = self.auto_format(sequence_output)
#
#         loss = 0.0
#         if node_labels is not None:
#             node_loss_fct = CrossEntropyLoss()
#             loss += node_loss_fct(node_logits.view(-1, self.num_labels), node_labels.view(-1))
#
#         if au_labels is not None:
#             link_loss_fct = MultiLabelCategoricalCrossEntropy()
#             loss += link_loss_fct(au_logits.squeeze(), au_labels)
#
#         if link_labels is not None:
#             link_loss_fct = MultiLabelCategoricalCrossEntropy()
#             loss += link_loss_fct(link_logits.squeeze(), link_labels)
#
#         if not return_dict:
#             output = (node_logits, au_logits, link_logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#
#         return AutoFormatV2Output(
#             loss=loss,
#             node_logits=node_logits,
#             au_logits=au_logits,
#             link_logits=link_logits,
#         )
#
#     @staticmethod
#     def mean_pooling(model_output: torch.FloatTensor, attention_mask: torch.FloatTensor) -> torch.FloatTensor:
#         first_embeddings, last_embeddings = model_output[1], model_output[-1]
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(first_embeddings.size()).float()
#         first_embeddings = torch.sum(first_embeddings * input_mask_expanded, 1)
#         last_embeddings = torch.sum(last_embeddings * input_mask_expanded, 1)
#         return (first_embeddings + last_embeddings) / (torch.clamp(input_mask_expanded.sum(1), min=1e-9) * 2)


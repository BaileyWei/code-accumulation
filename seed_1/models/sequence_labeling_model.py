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

# from seed.components.token_pair import TokenPairComponent
# from seed.components.outputs import TokenPairOutput
# from seed.layers.losses import MultiLabelCategoricalCrossEntropy


# class AlbertForTokenPairClassification(AlbertPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#
#     def __init__(self, config):
#         super().__init__(config)
#
#         config.update(config.task_specific_params)
#         self.config = config
#
#         # setattr(self, config.model_type, AutoModel(config))
#         self.albert = AlbertModel(config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = TokenPairComponent(config)
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
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], TokenPairOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.albert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = MultiLabelCategoricalCrossEntropy()
#             loss = loss_fct(logits, labels)
#
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#
#         return TokenPairOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
#
# class BertForTokenPairClassification(BertPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#
#     def __init__(self, config):
#         super().__init__(config)
#
#         config.update(config.task_specific_params)
#         self.config = config
#
#         self.bert = BertModel(config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = TokenPairComponent(config)
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
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], TokenPairOutput]:
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
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = MultiLabelCategoricalCrossEntropy()
#             loss = loss_fct(logits, labels)
#
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#
#         return TokenPairOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


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
        self.classifier = nn.Linear(config.hidden_size * (config.fusion_num + 1), config.num_labels)

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
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
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
            output_hidden_states=self.config.fusion,  # output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        intermediate_output = list()
        if self.config.fusion:
            for fusion_layer in self.config.fusion_layer:
                intermediate_output.append(outputs.hidden_states[fusion_layer])
            sequence_output = [sequence_output] + intermediate_output
            sequence_output = torch.cat(sequence_output, dim=-1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # outputs.hidden_states,
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
        self.classifier = TokenPairComponent(config)

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
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
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
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = MultiLabelCategoricalCrossEntropy()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# class DistilBertForTokenPairClassification(DistilBertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#
#         config.update(config.task_specific_params)
#         self.config = config
#
#         self.distilbert = DistilBertModel(config)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = TokenPairComponent(config)
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     def get_position_embeddings(self) -> nn.Embedding:
#         """
#         Returns the position embeddings
#         """
#         return self.distilbert.get_position_embeddings()
#
#     def resize_position_embeddings(self, new_num_position_embeddings: int):
#         """
#         Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
#         Arguments:
#             new_num_position_embeddings (`int`):
#                 The number of new position embedding matrix. If position embeddings are learned, increasing the size
#                 will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
#                 end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
#                 size will add correct vectors at the end following the position encoding algorithm, whereas reducing
#                 the size will remove vectors from the end.
#         """
#         self.distilbert.resize_position_embeddings(new_num_position_embeddings)
#
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], TokenPairOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.distilbert(
#             input_ids,
#             attention_mask=attention_mask,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = MultiLabelCategoricalCrossEntropy()
#             loss = loss_fct(logits, labels)
#
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output
#
#         return TokenPairOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
#
# class ElectraForTokenPairClassification(ElectraPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#
#         config.update(config.task_specific_params)
#         self.config = config
#
#         self.electra = ElectraModel(config)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = TokenPairComponent(config)
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
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], TokenPairOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         discriminator_hidden_states  = self.electra(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         discriminator_sequence_output  = discriminator_hidden_states[0]
#
#         discriminator_sequence_output  = self.dropout(discriminator_sequence_output )
#         logits = self.classifier(discriminator_sequence_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = MultiLabelCategoricalCrossEntropy()
#             loss = loss_fct(logits, labels)
#
#         if not return_dict:
#             output = (logits,) + discriminator_sequence_output[1:]
#             return ((loss,) + output) if loss is not None else output
#
#         return TokenPairOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=discriminator_hidden_states.hidden_states,
#             attentions=discriminator_hidden_states.attentions,
#         )


class RobertaForFusionTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        if config.task_specific_params:
            config.update(config.task_specific_params)
            config.fusion_num = len(config.fusion_layer)
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * (config.fusion_num + 1), config.num_labels)

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
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        input_features: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, (sequence_length + 1) * sequence_length / 2)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=self.config.fusion,  # output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        intermediate_output = list()
        if self.config.fusion:
            for fusion_layer in self.config.fusion_layer:
                intermediate_output.append(outputs.hidden_states[fusion_layer])
            sequence_output = [sequence_output] + intermediate_output
            sequence_output = torch.cat(sequence_output, dim=-1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )
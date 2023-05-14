# encoding: utf-8


from transformers import BertConfig, RobertaConfig


class BertNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertNerConfig, self).__init__(**kwargs)
        self.model_dropout = kwargs.get("model_dropout", 0.1)


class RobertaNerConfig(RobertaConfig):
    def __init__(self, **kwargs):
        super(RobertaConfig, self).__init__(**kwargs)
        self.model_dropout = kwargs.get("model_dropout", 0.1)

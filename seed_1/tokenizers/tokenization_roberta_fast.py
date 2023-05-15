import json
from tokenizers import pre_tokenizers, processors
from transformers import RobertaTokenizerFast

class PunctuationRobertaTokenizerFast(RobertaTokenizerFast):
    def __int__(
            self,
            vocab_file=None,
            merges_file=None,
            tokenizer_file=None,
            errors="replace",
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            add_prefix_space=False,
            trim_offsets=True,
            **kwargs
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file,
            errors,
            bos_token,
            eos_token,
            sep_token,
            cls_token,
            unk_token,
            pad_token,
            mask_token,
            add_prefix_space,
            trim_offsets,
            **kwargs
        )
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state["type"] == "ByteLevel":
            if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
                pre_tok_state["add_prefix_space"] = add_prefix_space
                self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        elif pre_tok_state["type"] == "Sequence":
            if pre_tok_state["pretokenizers"].get("add_prefix_space", add_prefix_space) != add_prefix_space:
                pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
                pre_tok_state["pretokenizers"][1]["add_prefix_space"] = add_prefix_space
                self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # The lists 'sep' and 'cls' must be cased in tuples for the object `post_processor_class`
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

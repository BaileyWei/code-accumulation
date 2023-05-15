import os, sys
import pickle
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
sys.path.append("/mnt/yaowx/experiment")

from seed.supervised_util import read_examples_from_file
from seed.models.token_pair_models import RobertaForTokenPairClassification
from seed.utils.data_collector import DataCollatorForEGPBaseline
from seed.utils.trainers import TrainerForLargeValidationDataset

train_data = read_examples_from_file('paddlepaddle/data/', mode='train')
val_data = read_examples_from_file('paddlepaddle/data/', mode='dev')
test_data = read_examples_from_file('paddlepaddle/data/', mode='test')

# 获取所有的类别标签
label_set = set()
for f in train_data:
    label_set.update(f.labels)
# max length & sliding window
max_length = 128
sliding_length = 32
# label map
labels = [l for l in label_set if l != 'O']
labels.sort()
label2id = {label:n for n, label in enumerate(labels)}
# tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)


def convert_tags_to_labels(tags):
    new_labels, previous_tag, start_idx = list(), None, None
    for idx, tag in enumerate(tags):
        if previous_tag == tag:
            continue
        elif start_idx:
            new_labels.append((start_idx, idx, previous_tag))
            start_idx, previous_tag = idx, tag
        else:
            start_idx, previous_tag = idx, tag
    new_labels.append((start_idx, idx, previous_tag))
    new_labels = [label for label in new_labels if label[-1] != "O"]
    return new_labels


def covert_word_ids_to_word_offset_mapping(word_ids):
    word_offset_mapping = dict()
    previous_idx, start_idx = None, None
    for n, idx in enumerate(word_ids):
        if idx is None and start_idx is None:
            continue
        elif idx is None and previous_idx:
            word_offset_mapping[previous_idx] = (start_idx, n)
            start_idx = n
            previous_idx = idx
        elif previous_idx != idx and start_idx is not None:
            word_offset_mapping[previous_idx] = (start_idx, n)
            start_idx = n
            previous_idx = idx
        elif previous_idx != idx:
            start_idx = n
            previous_idx = idx
    word_offset_mapping[previous_idx] = (start_idx, n)
    return word_offset_mapping


def split_into_short_samples(words, labels, tokenizer, max_length=128, sliding_length=32):
    max_raw_length = max_length - 2
    samples, left, right, last_run = list(), 0, max_raw_length - 1, False
    word_ids = tokenizer(words, add_special_tokens=False, is_split_into_words=True).word_ids()
    sequence_length = len(word_ids)
    word_offset_mapping = covert_word_ids_to_word_offset_mapping(word_ids)
    while not last_run:
        if right >= sequence_length + 2:
            last_run = True
        word_slices = [k for k, v in word_offset_mapping.items() if v[0] >= left and v[1] <= right]
        if not word_slices:
            continue  # a long and mean less words
        left_word, right_word = min(word_slices), max(word_slices)
        left, right = word_offset_mapping[left_word][0], word_offset_mapping[right_word][1]
        samples.append(
            (
                words[left_word: right_word + 1],
                [(s-left_word, e-left_word, t) for (s, e, t) in labels if left_word <= s and right_word >= e]
            )
        )
        if last_run:
            break
        left, right = left + sliding_length, left + sliding_length + max_raw_length - 1
    return samples


def pre_process(example):
    # convert tags to labels
    labels = convert_tags_to_labels(example.labels)
    # split into short samples
    samples = split_into_short_samples(example.words, labels, tokenizer, max_length=max_length,
                                       sliding_length=sliding_length)
    # convert words and labels into features
    features = list()
    for sample in samples:
        words, labels = sample
        tokens = tokenizer(words, is_split_into_words=True, max_length=max_length, padding="max_length")
        if len(tokens['input_ids']) > 128:
            print(1)
        word_offset_mapping = covert_word_ids_to_word_offset_mapping(tokens.word_ids())
        labels = [(word_offset_mapping[st][0], word_offset_mapping[ed-1][1], label2id[tp]) for (st, ed, tp) in labels]
        tokens['labels'] = labels
        tokens['word_ids'] = tokens.word_ids()
        features.append(tokens)

    return features


train_features = pickle.load(open('paddlepaddle/data/train_baseline.pkl', 'rb'))
val_features = pickle.load(open('paddlepaddle/data/val_baseline.pkl', 'rb'))
test_features = pickle.load(open('paddlepaddle/data/test_baseline.pkl', 'rb'))

train_dataset = Dataset.from_pandas(pd.DataFrame(train_features))
val_dataset = Dataset.from_pandas(pd.DataFrame(val_features))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_features))

def extract_span_entity(start_pred, end_pred):
    entity_span = list()
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                entity_span.append((i, i + j))
    return entity_span


def compute_metrics(eval_preds):
    result = dict()
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = (logits > 0).astype(int)
    predictions = np.where(labels == -100, 0, predictions)
    labels = np.where(labels == -100, 0, labels)
    result["tp"] = 0
    result["fp"] = 0
    result["fn"] = 0
    for prediction, label in zip(predictions, labels):
        pred_x, pred_y = np.nonzero(prediction)
        pred = [(i, j) for i, j in zip(pred_x.tolist(), pred_y.tolist())]
        gold_x, gold_y = np.nonzero(label)
        gold = [(i, j) for i, j in zip(gold_x.tolist(), gold_y.tolist())]
        pred, gold = set(pred), set(gold)
        result["tp"] += len(pred & gold)
        result["fp"] += len(pred - gold)
        result["fn"] += len(gold - pred)
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
    return metrics


training_args = TrainingArguments(
    output_dir="seed/ner_hf_baseline",
    overwrite_output_dir=True,
    prediction_loss_only=False,
    remove_unused_columns=False,
    evaluation_strategy="steps",
    eval_delay=10000,
    logging_steps=1000,
    save_steps=1000,
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=1,
    metric_for_best_model='eval_f1-score',
    greater_is_better=True,
    load_best_model_at_end=True,
    dataloader_num_workers=8,
)

# model = BertForUniversalNER.from_pretrained('bert-base-cased')
task_specific_params = {"tp_kernel": "EGP", "head_size": 64, "output_size": 66, "rope": True}
model = RobertaForTokenPairClassification.from_pretrained("roberta-base", task_specific_params=task_specific_params)

trainer = TrainerForLargeValidationDataset(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForEGPBaseline(
        tokenizer=tokenizer,
        max_length=max_length,
        model=model,
    ),
    compute_metrics=compute_metrics, #  compute_metrics
    gather_metrics=gather_metrics, # gather_metrics
)

trainer.train()
trainer.save_model()
trainer.state.save_state()

test_results = trainer.predict(test_dataset)
test_metrics = test_results.metrics
test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)

output_test_result_file = os.path.join(training_args.output_dir, "test_results.txt")
print("***** Test results *****")
with open(output_test_result_file, "w") as f:
    for key, value in sorted(test_metrics.items()):
        print(f"{key} = {value}\n")
        f.write(f"{key} = {value}\n")

print("Done!")
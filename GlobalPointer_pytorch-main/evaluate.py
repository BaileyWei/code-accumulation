"""
Date: 2021-06-11 13:54:00
LastEditors: GodK
LastEditTime: 2021-07-19 21:53:18
"""
import os
import config
import sys
import torch
import json
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, RobertaModel
from common.utils import DataMaker, MyDataset, MetricsCalculator
from models.GlobalPointer import GlobalPointer
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tokenization_roberta_fast import PunctuationRobertaTokenizerFast


config = config.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0
config["num_workers"] = 1


# for reproductivity
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=True, do_lower_case=False)
# tokenizer = PunctuationRobertaTokenizerFast(
#                                      vocab_file=r'./roberta/roberta-base/vocab.json',
#                                      merges_file=r'./roberta/roberta-base/merges.txt',
#                                      tokenizer_file=r'./roberta/roberta-base/tokenizer.json',)


metrics = MetricsCalculator()


def conll2clue(data):
    context = data['context']
    span_posLabel = data['span_posLabel']
    label = {}
    for span,type in span_posLabel.items():
        span_s = int(span.split(';')[0])
        span_e = int(span.split(';')[1])
        entity = ' '.join(context.split()[span_s:span_e+1])
        if label.get(type):
            type_dict = label.get(type)
            if type_dict.get(entity):
                type_dict.get(entity).append([span_s, span_e])
            else:
                type_dict[entity] = [[span_s, span_e]]
        else:
            label[type] = {entity: [[span_s, span_e]]}

    res_dict = {
        'text': context,
        'label': label,
        'span_posLabel': span_posLabel
    }

    return res_dict

def load_data(data_path, data_type="test", form=''):

    if data_type == "test":

        datas = []
        if form == 'conll03':
            lines = json.load(open(data_path, encoding="utf-8"))
            for i, line in enumerate(lines):
                item = {}
                item['id'] = line['idx']
                item["text"] = line["context"]

                datas.append(item)
            return datas
        else:
            with open(data_path, encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    datas.append(line)
            return datas
    if data_type == "train" or data_type == "valid":
        datas = []
        if form == 'conll03':
            lines = json.load(open(data_path, encoding="utf-8"))
            for line in lines:
                line = conll2clue(line)
                item = {}
                item["text"] = line["text"]
                item["entity_list"] = []
                item['span_posLabel'] = line['span_posLabel']
                for k, v in line['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entity_list"].append((start, end, k))
                datas.append(item)
            return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))


ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)


def data_generator(data_type="test"):
    """
    读取数据，生成DataLoader。
    """

    if data_type == "test":
        test_data_path = os.path.join(config["data_home"], config["exp_name"], config["test_data"])
        test_data = load_data(test_data_path, "test", form='conll03')
    if data_type == 'valid':
        test_data_path = os.path.join(config["data_home"], config["exp_name"], config["test_data_correction"])
        test_data = load_data(test_data_path, "valid", form='conll03')

    test_dataloader = DataLoader(MyDataset(test_data),
                                  batch_size=hyper_parameters["batch_size"],
                                  shuffle=True,
                                  num_workers=config["num_workers"],
                                  drop_last=False,
                                  collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                  )
    all_data = test_data

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer.tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))
    # assert max_tok_num <= hyper_parameters["max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

    data_maker = DataMaker(tokenizer)

    if data_type == "test":
        # test_inputs = data_maker.generate_inputs(test_data, max_seq_len, ent2id, data_type="test")
        test_dataloader = DataLoader(MyDataset(test_data),
                                     batch_size=hyper_parameters["batch_size"],
                                     shuffle=False,
                                     num_workers=config["num_workers"],
                                     drop_last=False,
                                     collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id,
                                                                                    data_type="test")
                                     )
        return test_dataloader
    if data_type == 'valid':
        test_dataloader = DataLoader(MyDataset(test_data),
                                     batch_size=hyper_parameters["batch_size"],
                                     shuffle=True,
                                     num_workers=config["num_workers"],
                                     drop_last=False,
                                     collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                     )

        return test_dataloader


def decode_ent(text, pred_matrix, tokenizer, threshold=0):
    # print(text)
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    id2ent = {id: ent for ent, id in ent2id.items()}
    pred_matrix = pred_matrix.cpu().numpy()
    ent_list = {}
    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = id2ent[ent_type_id]
        ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
        ent_text = text[ent_char_span[0]:ent_char_span[1]]

        ent_type_dict = ent_list.get(ent_type, {})
        ent_text_list = ent_type_dict.get(ent_text, [])
        ent_text_list.append(ent_char_span)
        ent_type_dict.update({ent_text: ent_text_list})
        ent_list.update({ent_type: ent_type_dict})
    # print(ent_list)
    return ent_list

def valid_step(batch_valid, model):
    # batch_token_type_ids,
    batch_samples, batch_input_ids, batch_attention_mask, batch_labels = batch_valid
    # batch_token_type_ids,
    batch_input_ids, batch_attention_mask, batch_labels = (batch_input_ids.to(device),
                                                           batch_attention_mask.to(device),
                                                         # batch_token_type_ids.to(device),
                                                           batch_labels.to(device)
                                                           )
    with torch.no_grad():
        logits = model(batch_input_ids, batch_attention_mask)


    R, T, X, Y, Z = metrics.get_evaluate_fpr(logits, batch_labels)

    return R, T, X, Y, Z


def valid(model, dataloader):
    model.eval()
    res = []
    total_R, total_T, total_X, total_Y, total_Z = set(), set(), 0., 0., 0.

    for batch_data in tqdm(dataloader, desc="Validating"):
        R, T, X, Y, Z = valid_step(batch_data, model)

        # total_R.add(R)
        # total_T.add(T)
        total_X += X
        total_Y += Y
        total_Z += Z
    avg_f1 = 2 * total_X / (total_Y + total_Z)
    avg_precision = total_X / total_Y
    avg_recall = total_X / total_Z
    print("******************************************")
    # f = open(model_state_dict_dir, 'w')
    print(f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}')
    print("******************************************")
    # if config["logger"] == "wandb":
    #     logger.log({"valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1})
    return avg_precision, avg_recall, avg_f1


def predict(dataloader, model, batch=True, metrics=False):
    res = []

    model.eval()
    if metrics:
        valid_precision, valid_recall, valid_f1= valid(model, dataloader)
    elif batch:
        for batch_data in dataloader:
            # batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, _ = batch_data
            batch_samples, batch_input_ids, batch_attention_mask,  _ = batch_data
            # batch_token_type_ids
            batch_input_ids, batch_attention_mask, = (batch_input_ids.to(device),
                                                                           batch_attention_mask.to(device),
                                                                           # batch_token_type_ids.to(device),
                                                                           )
            with torch.no_grad():
                batch_logits = model(batch_input_ids, batch_attention_mask)

            for ind in range(len(batch_samples)):
                gold_sample = batch_samples[ind]
                text = gold_sample["text"]
                text_id = gold_sample["id"]
                pred_matrix = batch_logits[ind]
                labels = decode_ent(text, pred_matrix, tokenizer)
                res.append({"id": text_id, "text": text, "label": labels})
    # else:
    #



        return res


def load_model():
    model_state_dir = config["model_state_dir"]
    model_state_list = sorted(filter(lambda x: "model_state" in x, os.listdir(model_state_dir)),
                              key=lambda x: int(x.split(".")[0].split("_")[-1]))
    last_k_model = config["last_k_model"]
    model_state_path = os.path.join(model_state_dir, model_state_list[-last_k_model])
    print(model_state_list[-last_k_model])

    encoder = RobertaModel.from_pretrained(config["bert_path"])
    model = GlobalPointer(encoder, ent_type_size, 64)
    model.load_state_dict(torch.load(model_state_path, map_location='cuda:0'), strict=True)
    model = model.to(device)

    return model, model_state_list[-last_k_model]


def evaluate(metrics=False, ):
    if metrics:
        test_dataloader = data_generator(data_type="valid")
    else:
        test_dataloader = data_generator(data_type="test")

    model, model_name = load_model()

    predict_res = predict(test_dataloader, model, metrics=metrics)
    if not metrics:
        if not os.path.exists(os.path.join(config["save_res_dir"], config["exp_name"])):
            os.mkdir(os.path.join(config["save_res_dir"], config["exp_name"]))
        save_path = os.path.join(config["save_res_dir"], config["exp_name"], f"{'_'.join(config['model_state_dir'].split('/')[-2:])}_{model_name}_predict_result.json")
        # json.dump(predict_res, open(save_path, "w", encoding="utf-8"), ensure_ascii=False)
        with open(save_path, "w", encoding="utf-8") as f:
            for item in predict_res:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    # for i in range(10):
    evaluate()

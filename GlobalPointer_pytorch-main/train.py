"""
Date: 2021-05-31 19:50:58
LastEditors: GodK
"""

import os
import config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel, RobertaModel
from transformers import AdamW
from common.utils import Preprocessor, multilabel_categorical_crossentropy, DataMaker, MyDataset, MetricsCalculator
from models.GlobalPointer import GlobalPointer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
# import wandb
from evaluate import evaluate
import time
from tokenization_roberta_fast import PunctuationRobertaTokenizerFast
from models.roberta_modeling import RobertaModelWithAlibi

import logging
# logging.basicConfig(format='%(asctime)s: %(message)s',
#                         level=logging.ERROR,
#                         filename='./runlogs.log',
#                         filemode='a',)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
config = config.train_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0
config["num_workers"] = 0


# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

if config["logger"] == "wandb" and config["run_type"] == "train":
    # init wandb
    wandb.init(project="GlobalPointer_" + config["exp_name"],
               config=hyper_parameters  # Initialize config
               )
    wandb.run.name = config["run_name"] + "_" + wandb.run.id

    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    model_state_dict_dir = os.path.join(config["path_to_save_model"], config["exp_name"],
                                        config['bert_path'],
                                        # time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime())
                                        f"b{hyper_parameters['batch_size']}_e{hyper_parameters['epochs']}_{hyper_parameters['lr']}"
                                        )
    epoch_loss_dir = os.path.join(model_state_dict_dir, 'epoch_loss')
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"] if 'legal' not in config["bert_path"] else 'bert-base-cased', add_special_tokens=True, do_lower_case=False)
# tokenizer = PunctuationRobertaTokenizerFast(
#                                      vocab_file=r'./roberta/roberta-base/vocab.json',
#                                      merges_file=r'./roberta/roberta-base/merges.txt',
#                                      tokenizer_file=r'./roberta/roberta-base/tokenizer.json',
#                                  )

def load_data(data_path, data_type="train", form=''):
    """读取数据集

    Args:
        data_path (str): 数据存放路径
        data_type (str, optional): 数据类型. Defaults to "train".

    Returns:
        (json): train和valid中一条数据格式：{"text":"","entity_list":[(start, end, label), (start, end, label)...]}
    """
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
        else:
            with open(data_path, encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    item = {}
                    item["text"] = line["text"]
                    item["entity_list"] = []
                    for k, v in line['label'].items():
                        for spans in v.values():
                            for start, end in spans:
                                item["entity_list"].append((start, end, k))
                    datas.append(item)
        return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))

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

ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)


def data_generator(data_type="train"):
    """
    读取数据，生成DataLoader。
    """

    if data_type == "train":
        train_data_path = os.path.join(config["data_home"], config["exp_name"], config["train_data"])
        train_data = load_data(train_data_path, "train", form='conll03' )  # form='conll03'
        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["valid_data"])
        valid_data = load_data(valid_data_path, "valid", form='conll03')
        test_data_path = os.path.join(config["data_home"], config["exp_name"], config["test_data"])
        test_data = load_data(test_data_path, "valid", form='conll03')
        test_data_path_c = os.path.join(config["data_home"], config["exp_name"], config["test_data_correction"])
        test_data_c = load_data(test_data_path_c, "valid", form='conll03')
    elif data_type == "valid":
        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["valid_data"])
        valid_data = load_data(valid_data_path, "valid",form='conll03')
        train_data = []

    all_data = train_data + valid_data + test_data + test_data_c

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer(sample["text"])["input_ids"]
        if len(tokens) > 128:
            print(tokens)
        max_tok_num = max(max_tok_num, len(tokens))
    # assert max_tok_num <= hyper_parameters[
    #     "max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

    data_maker = DataMaker(tokenizer)

    if data_type == "train":
        # train_inputs = data_maker.generate_inputs(train_data, max_seq_len, ent2id)
        # valid_inputs = data_maker.generate_inputs(valid_data, max_seq_len, ent2id)
        train_dataloader = DataLoader(MyDataset(train_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        test_dataloader = DataLoader(MyDataset(test_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        test_dataloader_c = DataLoader(MyDataset(test_data_c),
                                     batch_size=hyper_parameters["batch_size"],
                                     shuffle=True,
                                     num_workers=config["num_workers"],
                                     drop_last=False,
                                     collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                     )
        # for batch in train_dataloader:
        #     print(batch[1].shape)
        #     print(hyper_parameters["batch_size"])
        #     break
        return train_dataloader, valid_dataloader, test_dataloader, test_dataloader_c
    elif data_type == "valid":
        # valid_inputs = data_maker.generate_inputs(valid_data, max_seq_len, ent2id)
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        return valid_dataloader


metrics = MetricsCalculator()


def  train_step(batch_train, model, optimizer, criterion):
    # batch_input_ids:(batch_size, seq_len)    batch_labels:(batch_size, ent_type_size, seq_len, seq_len)
    # batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_train
    batch_samples, batch_input_ids, batch_attention_mask, batch_labels = batch_train

    batch_input_ids, batch_attention_mask, batch_labels = (batch_input_ids.to(device),
                                                           batch_attention_mask.to(device),
                                                                                 # batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                           )
    # T1 = time.perf_counter()
    # logits = model(batch_input_ids, batch_attention_mask )
    # T2 = time.perf_counter()
    # print('Time: %sms' % ((T2 - T1) * 1000))
    # T1 = time.perf_counter()
    logits = model(batch_input_ids, batch_attention_mask)
    # T2 = time.perf_counter()
    # print('Time: %sms' % ((T2 - T1) * 1000))
    loss = criterion(logits, batch_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# encoder = RobertaModel.from_pretrained('../pretrain_model/roberta-base/')
encoder = BertModel.from_pretrained(config['bert_path'])
# encoder = BertModel.from_pretrained('bert-base-cased')
model = GlobalPointer(encoder, ent_type_size, 64, alibi=False, RoPE=True)
model = model.to(device)

if config["logger"] == "wandb" and config["run_type"] == "train":
    wandb.watch(model)


def train(model, dataloader, epoch, optimizer):
    result = []
    model.train()

    # loss func
    def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    # scheduler
    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_dataloader) * rewarm_epoch_num,
                                                                         T_mult)
    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.
    for batch_ind, batch_data in pbar:

        loss = train_step(batch_data, model, optimizer, loss_fun)

        total_loss += loss

        avg_loss = total_loss / (batch_ind + 1)
        scheduler.step()

        pbar.set_description(f'Project:{config["exp_name"]}, Epoch: {epoch + 1}/{hyper_parameters["epochs"]}, Step: {batch_ind + 1}/{len(dataloader)}')
        pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

        if config["logger"] == "wandb" and batch_ind % config["log_interval"] == 0:
            logger.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })
        else:
            result.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })
    return result

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
        logits = model(batch_input_ids, batch_attention_mask,)
    R, T, X, Y, Z = metrics.get_evaluate_fpr(logits, batch_labels)

    return R, T, X, Y, Z


def valid(model, dataloader, test=False, correction=False):
    model.eval()
    desc = 'Validating'
    if test and correction:
        desc = 'testing_correction'
    elif test:
        desc = 'testing'
    total_R, total_T, total_X, total_Y, total_Z= 0., 0., 0., 0., 0.
    for batch_data in tqdm(dataloader, desc=desc):
        R, T, X, Y, Z = valid_step(batch_data, model)

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
    if config["logger"] == "wandb":
        logger.log({"valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1})
    return avg_precision, avg_recall, avg_f1


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    torch.cuda.empty_cache()
    if config["run_type"] == "train":
        train_dataloader, valid_dataloader, test_dataloader, test_dataloader_c = data_generator()

        # optimizer
        init_learning_rate = float(hyper_parameters["lr"])
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=init_learning_rate, eps=1e-8)
        f = open(epoch_loss_dir, 'w')
        f.write("dataset, epoch, precision, recall, f1\n")
        max_f1 = 0.
        loss = []
        for epoch in range(hyper_parameters["epochs"]):
            result = train(model, train_dataloader, epoch, optimizer)
            for _ in result:
                loss.append(_['train_loss'])

            valid_precision, valid_recall, valid_f1 = valid(model, valid_dataloader)
            test_precision, test_recall, test_f1 = valid(model, test_dataloader, test=True)
            test_precision_c, test_recall_c, test_f1_c = valid(model, test_dataloader_c, test=True, correction=True)

            f.write("valid, %d, %f, %f, %f\n" % (epoch, valid_precision, valid_recall, valid_f1,))
            f.write("test, %d, %f, %f, %f\n" % (epoch, test_precision, test_recall, test_f1,))
            f.write("test_correction, %d, %f, %f, %f\n" % (epoch, test_precision_c, test_recall_c, test_f1_c,))


            if valid_f1 > max_f1:
                max_f1 = valid_f1
                if valid_f1 > config["f1_2_save"]:  # save the best model
                    model_state_num = epoch
                    torch.save(model.state_dict(),
                               os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(model_state_num)))
            print(f"Best F1: {max_f1}")
            print("******************************************")
            if config["logger"] == "wandb":
                logger.log({"Best_F1": max_f1})
        p = plt.plot([i for i in range(len(loss))], loss)
        plt.savefig(r'{}/training_loss.png'.format(model_state_dict_dir))
    elif config["run_type"] == "eval":
        evaluate()

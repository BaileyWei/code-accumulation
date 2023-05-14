"""
Date: 2021-06-01 17:18:25
LastEditors: GodK
"""
import time

common = {
    # "exp_name": "cluener",
    "exp_name": "us_data5",
    "encoder": "BERT",
    "data_home": "./datasets",
    "bert_path": "/data/workspace/weiyubai/bert_fintune/model/lawtext/1/",  # bert-base-cased， bert-base-chinese
    # "bert_path": "roberta-base",  # bert-base-cased， bert-base-chinese, roberta-base

    "run_type": "train",  # train,eval
    "f1_2_save": 0.5,  # 存模型的最低f1值
    "logger": "default"  # wandb or default，default意味着只输出日志到控制台
}

# wandb的配置，只有在logger=wandb时生效。用于可视化训练过程
wandb_config = {
    "run_name": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    "log_interval": 10
}

train_config = {
    "train_data": "us_train.json",
    "valid_data": "us_dev.json",
    "test_data": "us_test.json",
    "test_data_correction": "us_test_correction.json",
    "ent2id": "ent2id.json",
    "path_to_save_model": "./outputs",  # 在logger不是wandb时生效
    "hyper_parameters": {
        "lr": 2e-5,
        "batch_size": 10,
        "epochs": 10,
        "seed": 2333,
        "max_seq_len": 512,
        "scheduler": "CAWR"
    }
}

eval_config = {
    "model_state_dir": "./outputs/us_data5/roberta-base/b10_e10_2e-05",  # 预测时注意填写模型路径（时间tag文件夹）
    "run_id": "",
    "last_k_model": 1,  # 取倒数第几个model_state
    "test_data_correction": "us_test_correction.json",
    "test_data": "us_test.json",
    "valid_data": "us_dev.json",
    "ent2id": "ent2id.json",
    "save_res_dir": "./results",
    "hyper_parameters": {
        "batch_size": 10,
        "max_seq_len": 512,
    }

}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------------------------
train_config["hyper_parameters"].update(**cawr_scheduler, **step_scheduler)
train_config = {**train_config, **common, **wandb_config}
eval_config = {**eval_config, **common}

import sys
import torch
import logging
from transformers import AdamW
from transformers import BertTokenizer
from transformers import BertForPreTraining
sys.path.append('/home/aidog/workspace/scarlett/Japan')
from bert_finetune.preprocess import Dataset

#logging configure
logging.basicConfig(level=logging.DEBUG, \
                    format = "%(asctime)s %(message)s", \
                    datefmt = "%Y-%m-%d %H:%M:%S")

#set path
train_data_path = '/data/scarlett/Japan/train'
test_data_path = '/data/scarlett/Japan/test'
model_path = '/data/wjb/pretrained_weight/bert-base-japanese'

#read pretrained model
model = BertForPreTraining.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

#read data
test_dataset = Dataset(test_data_path, model_path)
test_inputs = test_dataset.generate_dataset()

train_dataset = Dataset(train_data_path, model_path)
train_inputs = train_dataset.generate_dataset()


#set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#set optimizer
model.train()
optim = AdamW(model.parameters(), lr=5e-5)

epochs = 7
loss_all_train = 0
for epoch in range(epochs):
    for i,batch in enumerate(train_inputs):
        
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        next_sentence_label=next_sentence_label,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        loss_all_train += loss
        if i==0:
            logging.info('epoch:{}  batch:{}    trainloss:{}'.format(epoch, i, loss))
        if i!=0 and i%100==0:
            logging.info('epoch:{}  batch:{}    trainloss:{}'.format(epoch, i, loss_all_train/100))
            loss_all_train = 0
        
        if i!=0 and i%1000==0:
            with torch.no_grad():
                loss_all_eval = 0
                for i_eval,batch_eval in enumerate(test_inputs):
                    input_ids_eval = batch_eval['input_ids'].to(device)
                    token_type_ids_eval = batch_eval['token_type_ids'].to(device)
                    attention_mask_eval = batch_eval['attention_mask'].to(device)
                    next_sentence_label_eval = batch_eval['next_sentence_label'].to(device)
                    labels_eval = batch_eval['labels'].to(device)
                    outputs_eval = model(input_ids_eval, 
                                         attention_mask = attention_mask_eval,
                                         token_type_ids = token_type_ids_eval,
                                         next_sentence_label = next_sentence_label_eval,
                                         labels = labels_eval)
                    loss_eval = outputs_eval.loss
                    loss_all_eval += loss_eval
                logging.info('epoch:{}  batch:{}    evalloss:{}'.format(epoch, i, loss_all_eval/(i_eval+1)))


        loss.backward()
        # update parameters
        optim.step()
    model_path = '/data/scarlett/Japan/model/' + str(epoch)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logging.info('model_saved...')



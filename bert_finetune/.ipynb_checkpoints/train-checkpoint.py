import sys
import torch
from transformers import AdamW
from transformers import BertForPreTraining
sys.path.append('/home/aidog/workspace/scarlett/Japan')
from bert_finetune.preprocess import Dataset

train_data_path = '/data/scarlett/Japan/train'
test_data_path = '/data/scarlett/Japan/test'
model_path = '/data/wjb/pretrained_weight/bert-base-japanese'

#read pretrained model
model = BertForPreTraining.from_pretrained(model_path)

#read data
test_dataset = Dataset(test_data_path)
test_inputs = test_dataset.generate_dataset()

train_dataset = Dataset(train_data_path)
train_inputs = train_dataset.generate_dataset()


#set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#set optimizer
model.train()
optim = AdamW(model.parameters(), lr=5e-5)


epochs = 10
for epoch in range(epochs):
    for i,batch in enumerate(loader_train):
        #initializer calculated gradients
        optim.zero_grad()
        #pull all 



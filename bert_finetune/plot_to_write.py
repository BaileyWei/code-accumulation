#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

data = open('/home/aidog/workspace/scarlett/Japan/bert_finetune/nohup_mlm.out').readlines()
eval_loss_list = []
train_loss_list = []
train_loss_temp_list = []

eval_loss_mlm_list = []
train_loss_mlm_list = []
train_loss_mlm_temp_list = []

eval_loss_nsp_list = []
train_loss_nsp_list = []
train_loss_nsp_temp_list = []

for item in data:
    if ('trainloss' not in item) and ('evalloss' not in item):
        continue
    if 'evalloss' in item:
        train_loss_list.append(train_loss_temp_list)
        train_loss_temp_list  = []
        
        train_loss_mlm_list.append(train_loss_mlm_temp_list)
        train_loss_mlm_temp_list = []
        
        train_loss_nsp_list.append(train_loss_nsp_temp_list)
        train_loss_nsp_temp_list = []
        
        item_ = item.strip().split(' ')
        
        eval_loss = float(item_[4].split(':')[1])
        eval_loss_list.append(eval_loss)
        
        eval_loss_mlm = float(item_[5].split(':')[1])
        eval_loss_mlm_list.append(eval_loss_mlm)
        
        eval_loss_nsp = float(item_[6].split(':')[1])
        eval_loss_nsp_list.append(eval_loss_nsp)
        continue
    item_ = item.strip().split(' ')
    train_loss_temp_list.append(float(item_[4].split(':')[1]))
    train_loss_mlm_temp_list.append(float(item_[5].split(':')[1]))
    train_loss_nsp_temp_list.append(float(item_[6].split(':')[1]))
train_loss_list = [np.mean(item) for item in train_loss_list]
train_loss_mlm_list = [np.mean(item) for item in train_loss_mlm_list]
train_loss_nsp_list = [np.mean(item) for item in train_loss_nsp_list]
batch_list = [i for i in range(len(train_loss_list))]

plt.figure(figsize = (60,4))
plt.subplot(1,6,1)
plt.plot(batch_list, train_loss_list)
plt.title('train_loss')

plt.subplot(1,6,2)
plt.plot(batch_list,eval_loss_list)
plt.title('eval_loss')

plt.subplot(1,6,3)
plt.plot(batch_list, train_loss_mlm_list)
plt.title('train_mlm_loss')

plt.subplot(1,6,4)
plt.plot(batch_list,eval_loss_mlm_list)
plt.title('eval_mlm_loss')

plt.subplot(1,6,5)
plt.plot(batch_list, train_loss_nsp_list)
plt.title('train_nsp_loss')

plt.subplot(1,6,6)
plt.plot(batch_list,eval_loss_nsp_list)
plt.title('eval_nsp_loss')

plt.savefig('results_mlm.png')
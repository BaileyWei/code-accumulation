#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

data = open('/home/aidog/workspace/scarlett/summary/RM_experiments/environ/nohup.out').readlines()
eval_loss_list = []
eval_pre_list = []
train_loss_list = []
train_loss_temp_list = []

for item in data:
    if ('train_loss' not in item) and ('eval_loss' not in item):
        continue
    if 'eval_loss' in item:
        train_loss_list.append(train_loss_temp_list)
        train_loss_temp_list  = []
        item_ = item.strip().split(' ')
        eval_loss = float(item_[4].split(':')[1])
        eval_precision = float(item_[5].split(':')[1])
        eval_loss_list.append(eval_loss)
        eval_pre_list.append(eval_precision)
        continue
    item_ = item.strip().split(' ')
    train_loss_temp_list.append(float(item_[2].split(':')[1]))
train_loss_list = [np.mean(item) for item in train_loss_list]
batch_list = [i for i in range(63)]

plt.figure(figsize = (20,4))
plt.subplot(1,3,1)
plt.plot(batch_list, train_loss_list)
plt.title('train_loss')

plt.subplot(1,3,2)
plt.plot(batch_list,eval_loss_list)
plt.title('eval_loss')

plt.subplot(1,3,3)
plt.plot(batch_list, eval_pre_list)
plt.title('eval_precision')

plt.savefig('results.png')

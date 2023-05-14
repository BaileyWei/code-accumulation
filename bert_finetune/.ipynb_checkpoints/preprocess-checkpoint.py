#coding:utf-8
import torch
import random
from transformers import BertTokenizer


class Dataset:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

    def read(self):
        text = open(self.data_path,'r').readlines()
        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        return text, tokenizer

    def NSP(self, text):
        sentence_a = []
        sentence_b = []
        label = []

        for paragraph in text:
            sentences = [sentence for sentence in paragraph.split('。') if sentence != '']
            num_sentences = len(sentences)
            if num_sentences > 1:
                for i in range(10):
                    start = random.randint(0, num_sentences-2)
                    if random.random() >= 0.5:
                        # this is IsNextSentence
                        sentence_a.append(sentences[start])
                        sentence_b.append(sentences[start+1])
                        label.append(0)
                    else:
                        index_neg_text = random.randint(0, len(text)-1)
                        neg_text = text[index_neg_text]
                        neg_sentences = neg_text.split('。')
                        index_neg_sentence = random.randint(0, len(neg_sentences)-1)
                        # this is NotNextSentence
                        sentence_a.append(sentences[start])
                        sentence_b.append(neg_sentences[index_neg_sentence])
                        label.append(1)
        return {'sentence_a':sentence_a, 'sentence_b': sentence_b, 'label':label}


    def embedding(self, tokenizer, train):
        inputs = tokenizer(train['sentence_a'], train['sentence_b'], return_tensors= 'pt', max_length = 512, truncation = True, padding = 'max_length')
        inputs['next_sentence_label'] = torch.LongTensor([train['label']]).T
        inputs['labels'] = inputs.input_ids.detach().clone()
        return inputs

    def Mask(self, inputs):
        #create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(inputs.input_ids.shape)
        #create mask array
        mask_arr = (rand < 0.15) * (inputs.input_ids != 2) * (inputs.input_ids != 3) * (inputs.input_ids != 0)


        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 4

        return inputs

    def generate_dataset(self):
        text,tokenizer = self.read()
        train = self.NSP(text)
        inputs = self.Mask(self.embedding(tokenizer,train))
        train_dataset = torch.utils.data.DataLoader(inputs, batch_size=4,shuffle=True)
        return train_dataset


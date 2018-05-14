import os
import sys
import time
import torch
import spacy
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from hyphen import Hyphenator
from nltk.corpus import wordnet as wn
from utils.scorer import report_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class Extract_Features(object):

    def __init__(self):
        # self.embedding_model = spacy.load('es')
        self.embedding_model = spacy.load('es_core_news_md')
        self.h_es = Hyphenator('es_ES')

    def word_embeddings(self,target_word):
        embeddings = self.embedding_model(target_word)
        embeddings = torch.unsqueeze(torch.FloatTensor(embeddings.vector),dim=0)
        return embeddings   # 1 * 50

    def syllables(self,target_word):
        count_syl = 0
        for word in target_word.split():
            count_syl += len(self.h_es.syllables(word))
        return torch.from_numpy(np.array([[count_syl]])).type(torch.FloatTensor)   # 1 * 1

    def ambiguity(self,target_word):
        count_amb = 0
        for word in target_word.split():
            count_amb += len(wn.synsets(word,lang='spa'))
        return torch.from_numpy(np.array([[count_amb]])).type(torch.FloatTensor)   # 1 * 1

    def length(self,target_word):
        len_chars = len(target_word) / 6.2
        len_tokens = len(target_word.split(' '))
        return torch.from_numpy(np.array([[len_chars,len_tokens]])).type(torch.FloatTensor)   # 1 * 2

class NN(nn.Module):

    def __init__(self,n_features,n_hidden1,n_hidden2,n_hidden3,n_output):
        super(NN,self).__init__()
        self.hidden1 = nn.Linear(n_features,n_hidden1)
        # self.hidden2 = nn.Linear(n_hidden1,n_hidden2)
        # self.hidden3 = nn.Linear(n_hidden2,n_hidden3)
        self.output = nn.Linear(n_hidden1,n_output)

    def forward(self,input):
        Z1 = self.hidden1(input)
        A1 = F.relu(Z1)
        # Z2 = self.hidden2(A1)
        # A2 = F.relu(Z2)
        Z3 = self.output(A1)
        A3 = F.softmax(Z3,dim=1)
        # Z4 = self.output(A3)
        # A4 = F.softmax(Z4,dim=1)

        return A3

def combine(embeddings,count_syl,count_amb,length):
    
    features = torch.squeeze(torch.cat((embeddings,count_syl,count_amb,length),dim=1),dim=0)
    return features

def open_csv(file):
    path = os.getcwd() + '/datasets/spanish/' + file
    data = pd.read_csv(path,sep='\t',header=None,names=['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob'])
    return data

def save(net):
    torch.save(net,'net.pkl')

def extract_data(data,model):

    train_data_X = []
    train_data_Y = []

    for word in data['target_word']:
        a = model.word_embeddings(word)
        b = model.syllables(word)
        c = model.ambiguity(word)
        d = model.length(word)

        m = combine(a,b,c,d)

        train_data_X.append(m)

    for label in data['gold_label']:
        train_data_Y.append(label)

    train_data_Y = np.array(train_data_Y)
    train_Y = torch.from_numpy(train_data_Y)

    return torch.stack(train_data_X),train_Y

if __name__ == '__main__':

    model = Extract_Features()
    nn = NN(54,10,10,10,2)
    optimizer = torch.optim.Adam(nn.parameters(),lr=0.001,betas=(0.99,0.9))
    loss_func = torch.nn.CrossEntropyLoss()

    data_1 = open_csv('Spanish_Train.tsv')
    data_2 = open_csv('Spanish_Dev.tsv')
    data = pd.concat([data_1,data_2],ignore_index=True)

    train_data_X,train_Y = extract_data(data,model)

    losses = []

    for epoch in range(15000):

        train_X = Variable(train_data_X).type(torch.FloatTensor)
        Y = Variable(train_Y).type(torch.LongTensor)

        output = nn(train_X)

        loss = loss_func(output,Y)
        losses.append(loss.data[0])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(loss.data[0])

    save(nn)

    test_data = open_csv('Spanish_Test.tsv')

    test_data_Y = []

    for label in test_data['gold_label']:
        test_data_Y.append(label)

    test_data_X,test_Y = extract_data(test_data,model)

    test_X = Variable(test_data_X).type(torch.FloatTensor)

    predict = nn(test_X).data

    result,index = torch.max(predict,1)

    prediction = index.numpy().tolist()
    report_score(test_data_Y,prediction)

    plt.plot(losses)
    plt.show()











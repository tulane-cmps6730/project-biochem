# -*- coding: utf-8 -*-

import click
import sys
import os
import re
from collections import Counter
import itertools

import numpy as np
import pandas as pd
import re
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optimizer
import torchtext

from . import clf_path, config

@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
#@main.command('dl-data')
def dl_data():
    """
    Download training/testing data.
    """
    # data_url = config.get('data', 'url')
    # data_file = config.get('data', 'file')
    # print('downloading from %s to %s' % (data_url, data_file))
    # r = requests.get(data_url)
    # with open(data_file, 'wt') as f:
    #     f.write(r.text)
    pass
    
@main.command('dl-data')
def data2df():
    return pd.read_excel('./nlp/biggeerdata-cleaned.xlsx')#pd.read_csv(config.get('data', 'file'))

@main.command('stats')
def stats():
    """
    Read the data files and print interesting statistics.
    """
    df = data2df()
    print('%d rows' % len(df))
    print('label counts:')
    print(df.Sentiment.value_counts())    

@main.command('train')
def train():
    """
    Train a classifier and save it.
    """
    # (1) Read the data...
    df = data2df()    
    torch.manual_seed(32)
    # (2) Create classifier and vectorizer.
    
    classific = [x for x in df['Sentiment']]
    unclean_sentences = [x for x in df['Breakdown']]
    
    # decided to use stratify to ensure even representations
    x_train,x_test,y_train,y_test = train_test_split(unclean_sentences ,classific,stratify=classific, random_state=32)
    real_train, real_test, real_vocab, real_train_words, real_test_words = final_tokenizing_funct(x_train, x_test)
    
    
    sents_as_tokens = [[s for s in sent.split(' ') if s != ''] for sent in unclean_sentences]
    longest_sent = find_max(sents_as_tokens)
    
    
    # my long prep steps bc why not

    x_train_pad = pad_for_model(real_train, longest_sent)
    x_test_pad = pad_for_model(real_test, longest_sent)

    # to prevent errors later
    x_train_pad = np.array(x_train_pad)
    x_test_pad = np.array(x_test_pad)
    y_train_pad = np.array(y_train)
    y_test_pad = np.array(y_test)

    #shortening these so I can send them through the model in equal chunks

    x_train_pad = x_train_pad[:126]
    x_testtt = x_test_pad[:42]

    x_test_pad = x_testtt[21:]
    x_valid_pad = x_testtt[:21]


    y_train_pad = y_train_pad[:126]

    y_testtt = y_test_pad[:42]
    y_test_pad = y_testtt[21:]
    y_valid_pad = y_testtt[:21]
    
    training_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train_pad))
    validation_data = TensorDataset(torch.from_numpy(x_valid_pad), torch.from_numpy(y_valid_pad))

    batch_size = 3

    training_loader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(validation_data, shuffle=True, batch_size=batch_size)
    
    it = iter(training_loader)
    sample_x, sample_y = next(it)
    
    num_layers = 2
    vocab_size = len(real_vocab) + 1
    embedding_size = 64
    output_size = 2
    hidden_size = 256
    
    model = SentimentReviewRNN(num_layers, vocab_size, hidden_size, embedding_size, output_size, drop_prob=0.5)

    lr=0.001 #yuh learning rate

    criterion = nn.BCELoss() #choice of loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    testing_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test_pad))
    testing_loader = DataLoader(testing_data, shuffle=True, batch_size=batch_size)
    
    params_grid = {'learning_rate': [0.001, 0.01, 0.1],'num_epochs': [5, 10, 15]}
    
    print(train_sentRNN(params_grid, model, training_loader, valid_loader, testing_loader))

    
    
    
    
    # clf = LogisticRegression(max_iter=1000, C=1, class_weight='balanced')         
    # vec = CountVectorizer(min_df=5, ngram_range=(1,3), binary=True, stop_words='english')
    # X = vec.fit_transform(df.title)
    # y = df.partisan.values
    # # (3) do cross-validation and print out validation metrics
    # # (classification_report)
    # do_cross_validation(clf, X, y)
    # # (4) Finally, train on ALL data one final time and
    # # train. Save the classifier to disk.
    # clf.fit(X, y)
    # pickle.dump((clf, vec), open(clf_path, 'wb'))
    # top_coef(clf, vec)
    
    


def clean_up_word(val):

    val = re.sub(r"[^\w\s]", '', val)
    val = re.sub(r"\s+", '', val)
    val = re.sub(r"\d", '', val)

    #more for good measure in case i dont understand re
    val = val.strip('.!,-\()')

    return val

def final_tokenizing_funct(x_train, x_test):

    vocab_list = []
    train_clean_sents_butwords = []
    test_clean_sents_butwords = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        temp = []
        for word in sent.lower().split():
            word = clean_up_word(word)
            if word not in stop_words and word != '':
                vocab_list.append(word)
                temp.append(word)
        train_clean_sents_butwords.append(temp)
    for sent in x_test:
        temp = []
        for word in sent.lower().split():
            word = clean_up_word(word)
            if word not in stop_words and word != '':
                temp.append(word)
        test_clean_sents_butwords.append(temp)   
            
    the_corpus = Counter(vocab_list)
    corpus = sorted(the_corpus,key=the_corpus.get,reverse=True)[:1000]
    onehot_dict = {w:i+1 for i,w in enumerate(corpus)}


    train_sents_clean = []
    test_sents_clean = []
    for sent in x_train:
          train_sents_clean.append([onehot_dict[clean_up_word(word)] for word in sent.lower().split()
                                     if clean_up_word(word) in onehot_dict.keys()])
    for sent in x_test:
            test_sents_clean.append([onehot_dict[clean_up_word(word)] for word in sent.lower().split()
                                    if clean_up_word(word) in onehot_dict.keys()])


    return train_sents_clean, test_sents_clean, onehot_dict, train_clean_sents_butwords, test_clean_sents_butwords

def find_max(sentences):
  return max(len(s) for s in sentences)


def pad_for_model(sentences, longest_sentence):
    features = np.zeros((len(sentences), longest_sentence),dtype=int)

    for index, review in enumerate(sentences):

        if len(review) != 0:
            features[index, -len(review):] = np.array(review)[:longest_sentence]

    return features

# def do_cross_validation(clf, X, y):
#     all_preds = np.zeros(len(y))
#     for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X,y):
#         clf.fit(X[train], y[train])
#         all_preds[test] = clf.predict(X[test])
#     print(classification_report(y, all_preds))    

# def top_coef(clf, vec, labels=['liberal', 'conservative'], n=10):
#     feats = np.array(vec.get_feature_names_out())
#     print('top coef for %s' % labels[1])
#     for i in np.argsort(clf.coef_[0])[::-1][:n]:
#         print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))
#     print('\n\ntop coef for %s' % labels[0])
#     for i in np.argsort(clf.coef_[0])[:n]:
#         print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))
        
class SentimentReviewRNN(nn.Module):
    #def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
    def __init__(self, num_layers, vocab_size, hidden_size, embedding_size, output_size, drop_prob=0.5):
        super(SentimentReviewRNN,self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size,hidden_size=self.hidden_size,
                           num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        # x = the batch of input sequences
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        out = self.dropout(lstm_out)
        out = self.fc(out)

        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):

        h0 = torch.zeros((self.num_layers,batch_size,self.hidden_size))
        c0 = torch.zeros((self.num_layers,batch_size,self.hidden_size))
        hidden = (h0,c0)
        return hidden

def test_sentReviewRNN(model, data_loader):
  testing_loss = 0.0
  num_correct = 0
  total = 0

  predictions = []
  true_labels = []

  model.eval()


  with torch.no_grad():
    for inputs, labels in data_loader:

      batch_size = labels.size(0)

      test_h = model.init_hidden(batch_size)
      test_h = tuple([each.data for each in test_h])

      output, test_h = model(inputs, test_h)
     # loss = loss_funct(output.squeeze(), labels.float())
      #testing_loss += loss.item()

      prediction = torch.round(output)
      print(prediction)

      num_correct += (prediction == labels).sum().item()
      total += labels.size(0)

      predictions.extend(prediction.numpy())
      true_labels.extend(labels.numpy())


  #testing_loss = testing_loss / len(data_loader)
  accuracy = num_correct / total
  predictions = [int(x) for x in predictions]
  print(predictions)
  print(true_labels)


  return predictions, true_labels,accuracy

def train_sentRNN(param_grid, model_type, training_load, valid_load, test_load):
  clip = 5 #attempt to avoid exploding gradient
  #epochs = 10
  valid_loss_min = np.Inf
  batch_size = 3

  epoch_train_loss = []
  epoch_valid_loss = []
  epoch_train_acc = []
  epoch_valid_acc = []
  
  #lr=0.001 #yuh learning rate
  criterion = nn.BCELoss() #choice of loss function
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
  
  best_params = None
  best_accuracy = 0.0
  param_combinations = list(itertools.product(*param_grid.values()))

  for p in param_combinations:
    model = model_type
    optimizer = torch.optim.Adam(model.parameters(), lr=p[0])
    for epoch in range(p[1]):
        train_losses = []
        train_acc = 0.0
        model.train()

        h = model.init_hidden(batch_size)
        for inputs, labels in training_load:
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)

            loss = model.criterion(output.squeeze(), labels.float())
            loss.backward()
            train_losses.append(loss.item())
            accuracy = acc(output, labels)
            train_acc += accuracy

            #`clip_grad_norm` - prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()


            val_h = model.init_hidden(batch_size)


        val_losses = []
        val_acc = 0.0
        model.eval()
        for inputs, labels in valid_load:
                val_h = tuple([each.data for each in val_h])


                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

                accuracy = acc(output,labels)
                val_acc += accuracy

        epoch_tr_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_tr_acc = train_acc/len(training_load.dataset)
        epoch_val_acc = val_acc/len(valid_load.dataset)
        #epoch_train_loss.append(epoch_tr_loss)
        #epoch_valid_loss.append(epoch_val_loss)
        epoch_train_acc.append(epoch_tr_acc)
        epoch_valid_acc.append(epoch_val_acc)


    pred, true, test_acc = test_sentReviewRNN(model, testing_load)
    if test_acc > best_accuracy:
      best_accuracy = test_acc
      best_params = p


  return best_params, best_accuracy


if __name__ == "__main__":
    sys.exit(main())

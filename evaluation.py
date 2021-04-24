from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
import random
import torch
import copy





class Plotter:
    '''
    Plotter class is used for qualitative analysis. 
    It allows to plot the attention weights associated with an input and
    a model including attention, and to plot the prediction scores of each classes.
    
    It takes as inputs a source vocabulary, a target vocabulary and a model.
    
    Functions:
    - numericalize: transforms an iterable into a numeric tensor with respect to
        source vocabulary
    - plot_attn_probas: plots attention weights and preidction probabilities of an input
        based on the model outputs
    '''
    def __init__(self, src_vocab, tgt_vocab, model):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.model = model
        
    def numericalize(self, name):
        '''
        Transforms an iterable into a numeric tensor with respect to source vocabulary
        '''
        return torch.LongTensor([self.src_vocab.stoi(token) 
                                 for token in name])
    
    def plot_attn_probas(self, name):
        '''
        Plotting attention weights associated of the prediction of a model based 
        on an input, and plotting the prediction probabilities.
        It takes a string as an input.
        '''
        # numericalize the string
        num_name = self.numericalize(name)
        src = num_name.unsqueeze(1)
        src_len = torch.LongTensor([len(name)])
        
        # prediction
        with torch.no_grad():
            pred, attn = self.model(src, src_len)
        pred = pred.view(-1)
        
        # get probas from the output of the model (softmax)
        mask = torch.BoolTensor([True] * 3 + [False] * (len(self.tgt_vocab) - 3))
        pred_probas = torch.nn.functional.softmax(pred.masked_fill(mask, float('-inf')), 
                                                  dim=0)[3:].tolist()
        pred_idx = pred_probas.index(max(pred_probas))
        pred_token = self.tgt_vocab[pred_idx]
        
        # attention weights
        attn = attn[0].numpy()[0][0].reshape(1, -1)
        
        # x_labels : include <cls> if self attention
        if attn.shape[1] == len(name):
            xticklabels=[char for char in name]
        else:
            xticklabels=['<cls>']+[char for char in name]
        
        # plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4), sharey=False, 
                                       gridspec_kw={'width_ratios': [2, 2]})
        fig.tight_layout(pad=3.0)
        # attention weights
        sns.heatmap(attn, xticklabels=xticklabels, 
            cmap=matplotlib.cm.Blues, yticklabels=False, cbar=True, ax=ax1)
        # probabilities
        ax2.barh(self.tgt_vocab[3:], pred_probas, height=0.5)
        


def prediction(model, loader):
    '''
    Make predictions with respect to a model. 
    Takes as inputs the model and a dataloader on which prediction will be made.
    Returns the list of admissible outputs and the predicted output.
    '''
    model.eval()
    admissible_tgt = list()
    pred_tgt = list()
    for (src, src_length), tgt, adm_tgt in loader:
        with torch.no_grad():
            pred, _ = model(src, src_length)
            pred_tgt.extend(pred.topk(k=1, dim=1)[1].squeeze(1).tolist())
            admissible_tgt.extend(adm_tgt)

        admissible_tgt = [item for item in admissible_tgt]
        pred_tgt = [item for item in pred_tgt]
    
    return admissible_tgt, pred_tgt


def custom_accuracy(y_true, y_pred):
    '''
    Custom accuracy score taking into account multiple outputs possible
    '''
    return np.mean([1 if pred in true else 0
                        for pred, true in zip(y_pred, y_true)])


def evaluate(y_true, y_pred):
    return round(custom_accuracy(y_true, y_pred), 5)
    

def unique_target(y_true, y_pred):
    '''
    Retreive a unique target from the list of admissible targets and prediction
    '''
    true_tgt = list()
    for i in range(len(y_pred)):
        if y_pred[i] in y_true[i]:
            true_tgt.append(y_pred[i])
        else:
            true_tgt.append(max(y_true[i])) # max of indices so not class 'unknown' (idx=2)
    return true_tgt
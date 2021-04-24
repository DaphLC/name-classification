from collections import Counter

import numpy as np
import random
import torch
import copy


class Vocab:
    '''
    Construct a vocabulary initialized with the following tokens and completed from a list of inputs:
    - <pad>: padding 
    - <cls>: special classification token
    - <unk>: unknown character 
    
    The class contains a vocabulary (token to index), its associated reverse vocabulary (index to token),
    and the frequences of each tokens in the inputs used to construct the vocabulary.
    
    Fonctions:
    - add_tokens_from_iterable: scans an iterable to add its tokens to the vocabulary
    - add_token: updates vocabulary and reverse vocabulary with the token if not already in it 
    - itos (index to string): returns the token (string) associated with the index (position) in the vocabulary
    - stoi (string to index): returns the position (index) in the vocabulary of the token (string)
    - __len__: length of the vocabulary (number of distinct tokens)
    - __repr__: representation of the vocabulary
    - __getitem__: call function
    '''
    def __init__(self):
        # vocab starts with tokens: padding, class, unknown
        self.vocab = ['<pad>', '<cls>', '<unk>']
        self.reverse_vocab = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.freqs = Counter()
        
        # unknown token index in vocabulary
        self._unk_idx = self.reverse_vocab['<unk>']
    
    def add_tokens_from_iterable(self, iterable):
        for token in iterable:
            self.add_token(token)
                
    def add_token(self, token):
        self.freqs[token] += 1
        if token not in self.vocab:
            self.vocab.append(token)
            self.reverse_vocab[token] = len(self.vocab)-1
    
    def itos(self, position):
        '''
        Index to String
        '''
        if position >= len(self.vocab):
            return '<unk>'
        return self.vocab[position]
        
    def stoi(self, token): 
        '''
        String to Index
        '''
        return self.reverse_vocab.get(token, self._unk_idx)
        
    def __len__(self):
        return len(self.vocab)
    
    def __repr__(self):
        s = ",\n ".join(self.vocab)
        s = f'[{s}]'
        return s
    
    def __getitem__(self, index):
        return self.vocab[index]
    
    
    
class Dataset:
    '''
    Construct a dataset from a dictionary of names associated with origins, 
    a dictionary of admissible origins by name and optional vocabularies (names and origins).
    If no vocabulary is set, both names and origin vocabulary are updated with respect to the dataset.
    
    Fonctions:
    - numericalize: transforms an iterable into a numeric tensor 
        (i.e. each token of the iterable is replaced by its position in the vocabulary)
    - __len__: length of the dataset (number of unique triplets 'name', 'origin', 'admissible origins')
    - __getitem__: call function returns the numericalized item
    '''
    def __init__(self, names, names_to_origin, 
                 names_vocab=None, origin_vocab=None):
        # dataset consists of a name, its origin and all its admissible origins
        self.data = [[n['Name'], n['Origin'], names_to_origin[n['Name']]] 
                     for n in names]
        
        if names_vocab==None:
            self.names_vocab = Vocab()
            self.origin_vocab = Vocab()
            for name, origin, _ in self.data:
                self.names_vocab.add_tokens_from_iterable(name)
                self.origin_vocab.add_token(origin)
        else:
            assert origin_vocab is not None
            self.names_vocab = names_vocab
            self.origin_vocab = origin_vocab

    @staticmethod
    def numericalize(iterable, vocab):
        '''
        Transforms an iterable into a numeric tensor
        (i.e. each token of the iterable is replaced by its position in the vocabulary)
        '''
        return torch.LongTensor([vocab.stoi(token) 
                                 for token in iterable])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        # if item is the position in the dataset
        if isinstance(item, (int, tuple, slice)):
            src, tgt, admissible_tgt = self.data[item]
            src = self.numericalize(src, self.names_vocab)
            tgt = self.numericalize([tgt], self.origin_vocab)
            admissible_tgt = self.numericalize(admissible_tgt, self.origin_vocab) 
            return src, tgt, admissible_tgt
        
        # if item is the 
        if isinstance(item, str):
            return self.numericalize(item, self.names_vocab)
        
        # if neither of above
        raise NotImplementedError(f'{type(item).__name__} is not a valid item')
        
          
        
def pad(inputs):
    '''
    Custom padding function used in 'collate_fn' keeping in memory the initial lengths of the inputs.
    
    '''
    init_length = [len(i) for i in inputs]
    pad_inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value=0)
    
    return pad_inputs, init_length



def collate_fn(batch):
    '''
    Custom collate function used in the Dataloader to create batchs.
    Returns inputs with padding, inputs initial lengths, outputs and admissible_outputs.
    '''
    inputs, init_length = pad([b[0] for b in batch])
    outputs = torch.cat([b[1] for b in batch], dim=0)
    admissible_outputs = [b[2].tolist() for b in batch]
    
    return (inputs, init_length), outputs, admissible_outputs




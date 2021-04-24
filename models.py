import numpy as np
import gensim
import random
import torch
import copy




class LSTMWithAttn(torch.nn.Module):
    '''
    LSTM model with a layer of self-attention
    
    Inputs: 
    - vocab: source vocabulary
    - hidden_dim: dimension of hidden layers
    - n_classes: number of output classes
    - n_layers: number of layers
    - bidirectional: whether the LSTM is bidirectional
    - char2vec: Char2Vec embeddings
    - freeze: whether to freeze the char2vec embeddings
    
    Fonctions:
    - char2vec_to_tensor: matchs vocabulary tokens with char2vec embeddings 
    - repr_dim: layer dimension after lstm
    - forward: forward pass through the model
    '''
    def __init__(self, vocab, hidden_dim, n_classes, n_layers,
                 bidirectional=False, char2vec=None, freeze=False):
        super().__init__()
        
        self.vocab = vocab
        self.pad_idx = self.vocab.stoi('<pad>')
        self.cls_idx = self.vocab.stoi('<cls>')
        
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.char2vec = char2vec
        
        # for manual self-attention 
        query = torch.nn.Parameter(torch.ones(1, 1, self.repr_dim))
        self.register_parameter('query', query)
        
        # embeddings: pretrained or char2vec
        if char2vec is None:
            self.embedding = torch.nn.Embedding(
                num_embeddings=len(self.vocab),
                embedding_dim=self.hidden_dim,
                padding_idx=self.pad_idx)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(
                self.char2vec_to_tensor(self.vocab, 
                                        self.char2vec, 
                                        self.hidden_dim), 
                freeze=freeze, 
                padding_idx=self.pad_idx
            )
        
        self.lstm = torch.nn.LSTM(input_size=self.hidden_dim, 
                                hidden_size=self.hidden_dim,
                                num_layers=self.n_layers,
                                bidirectional=self.bidirectional)
        
        self.out = torch.nn.Linear(in_features=self.repr_dim, 
                                   out_features=self.n_classes)
        
    @staticmethod
    def char2vec_to_tensor(vocab, char2vec, hidden_dim):
        '''
        Matchs vocabulary tokens with char2vec embeddings 
        '''
        embeddings = list()
        unknown_char = list()
        for idx, char in enumerate(vocab.vocab):
            if char in char2vec.wv:
                embeddings.append(char2vec.wv[char]) 
            else:
                # sanity check
                unknown_char.append(char)
                embeddings.append(np.random.uniform(low=-1, high=1, 
                                                    size=hidden_dim))
        print("The following tokens don't have pre-trained vectors and are initialized randomly: ", 
              unknown_char)
        return torch.FloatTensor(embeddings)
    
    @property
    def repr_dim(self):
        return self.hidden_dim * (1 + self.bidirectional)
        
    def forward(self, src, src_lengths):
        '''
        Forward pass through the model: embedding, padding, lstm, attention, out.
        '''
        
        padding_mask = src.eq(self.pad_idx).transpose(0, 1).unsqueeze(1)
        
        seq_len, batch_size = src.shape
       
        src = self.embedding(src)
        
        src = torch.nn.utils.rnn.pack_padded_sequence(src, src_lengths, 
                                                      enforce_sorted=False)
        
        outputs, (hx, cx) = self.lstm(src)
        
        # unpack src and permute
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        
        # expand query to batch size, -1 for unchanged dim
        query = self.query.expand(batch_size, -1, -1)
            
        # multiply src & query: self-attention
        attn_weights = torch.bmm(query, outputs.permute(1, 2, 0))
        
        # mask the padding
        attn_weights = attn_weights.masked_fill(padding_mask, float('-inf'))
        
        # self explanatory
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # matrix multiplication
        context = torch.bmm(attn_weights, outputs.transpose(0, 1)).squeeze(1)
        
        return self.out(context), [attn_weights]
    
    
    
    
class LSTM(torch.nn.Module):
    '''
    LSTM model with no layer of self-attention
    
    Inputs: 
    - vocab: source vocabulary
    - hidden_dim: dimension of hidden layers
    - n_classes: number of output classes
    - n_layers: number of layers
    - bidirectional: whether the LSTM is bidirectional
    - char2vec: Char2Vec embeddings
    - freeze: whether to freeze the char2vec embeddings
    
    Fonctions:
    - char2vec_to_tensor: matchs vocabulary tokens with char2vec embeddings 
    - repr_dim: layer dimension after lstm
    - forward: forward pass through the model
    '''
    def __init__(self, vocab, hidden_dim, n_classes, n_layers,
                 bidirectional=False, char2vec=None, freeze=False):
        super().__init__()
        
        self.vocab = vocab
        self.pad_idx = self.vocab.stoi('<pad>')
        self.cls_idx = self.vocab.stoi('<cls>')
        
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.char2vec = char2vec
        
        # embeddings: pretrained or char2vec
        if char2vec is None:
            self.embedding = torch.nn.Embedding(
                num_embeddings=len(self.vocab),
                embedding_dim=self.hidden_dim,
                padding_idx=self.pad_idx)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(
                self.char2vec_to_tensor(self.vocab, 
                                        self.char2vec, 
                                        self.hidden_dim), 
                freeze=freeze, 
                padding_idx=self.pad_idx
            )
            
        self.lstm = torch.nn.LSTM(input_size=self.hidden_dim, 
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.n_layers,
                                  bidirectional=self.bidirectional)
        
        self.out = torch.nn.Linear(in_features=self.repr_dim, 
                                   out_features=self.n_classes)
        
    @staticmethod
    def char2vec_to_tensor(vocab, char2vec, hidden_dim):
        '''
        Matchs vocabulary tokens with char2vec embeddings 
        '''
        embeddings = list()
        unknown_char = list()
        for idx, char in enumerate(vocab.vocab):
            if char in char2vec.wv:
                embeddings.append(char2vec.wv[char]) 
            else:
                # sanity check
                unknown_char.append(char)
                embeddings.append(np.random.uniform(low=-1, high=1, 
                                                    size=hidden_dim))
        print("The following tokens don't have pre-trained vectors and are initialized randomly: ", 
              unknown_char)
        return torch.FloatTensor(embeddings)
    
    @property
    def repr_dim(self):
        return self.hidden_dim * self.n_layers * (1 + self.bidirectional)
        
    def forward(self, src, src_lengths):
        '''
        Forward pass through the model: embedding, padding, lstm, out.
        '''
        attns = list()
       
        src = self.embedding(src)
        
        src = torch.nn.utils.rnn.pack_padded_sequence(src, src_lengths, 
                                                      enforce_sorted=False)
        src, (hx, cx) = self.lstm(src)
        
        # last layer in both directions (bidirectionnel)
        hx = hx.transpose(0, 1).contiguous().view(-1, self.repr_dim)
        
        return self.out(hx), attns
    
    
    

    
class SelfAttention(torch.nn.Module):
    '''
    Self Attention model with positional embedding
    
    Inputs: 
    - vocab: source vocabulary
    - hidden_dim: dimension of hidden layers
    - n_classes: number of output classes
    - num_heads: number of heads of attention
    - n_layers: number of layers
    - char2vec: Char2Vec embeddings
    - freeze: whether to freeze the char2vec embeddings
    
    Fonctions:
    - char2vec_to_tensor: matchs vocabulary tokens with char2vec embeddings 
    - forward: forward pass through the model
    '''
    def __init__(self, vocab, hidden_dim, n_classes, num_heads, n_layers,
                 char2vec=None, freeze=False, pos_enc = 10):
        super().__init__()
        
        self.vocab = vocab
        self.pad_idx = self.vocab.stoi('<pad>')
        self.cls_idx = self.vocab.stoi('<cls>')
        
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.num_heads = num_heads # number of layers of attention on each token
        self.n_layers = n_layers
        
        self.char2vec = char2vec

        # sanity check
        assert self.hidden_dim % self.num_heads == 0
        
        # embeddings: pretrained or char2vec
        if char2vec is None:
            self.embedding = torch.nn.Embedding(
                num_embeddings=len(self.vocab),
                embedding_dim=self.hidden_dim,
                padding_idx=self.pad_idx)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(
                self.char2vec_to_tensor(self.vocab, 
                                        self.char2vec, 
                                        self.hidden_dim), 
                freeze=freeze, 
                padding_idx=self.pad_idx
            )
        
        # max len of input = 50
        self.positional_embedding = torch.nn.Embedding(50, pos_enc)
        
        self.attentions = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(embed_dim=self.hidden_dim+pos_enc,
                                        num_heads=self.num_heads) 
            for i in range(self.n_layers)])
        
        self.out = torch.nn.Linear(in_features=self.hidden_dim+pos_enc, 
                                   out_features=self.n_classes)
        
    @staticmethod
    def char2vec_to_tensor(vocab, char2vec, hidden_dim):
        '''
        Matchs vocabulary tokens with char2vec embeddings 
        '''
        embeddings = list()
        unknown_char = list()
        for idx, char in enumerate(vocab.vocab):
            if char in char2vec.wv:
                embeddings.append(char2vec.wv[char]) 
            else:
                # sanity check
                unknown_char.append(char)
                embeddings.append(np.random.uniform(low=-1, high=1, 
                                                    size=hidden_dim))
        print("The following tokens don't have pre-trained vectors and are initialized randomly: ", 
              unknown_char)
        return torch.FloatTensor(embeddings)
        
    def forward(self, src, *args, **kwargs):
        '''
        Forward pass through the model: embedding, positional embedding, self-attention with
        padding, out.
        '''
        # concatenate token 'cls' and src
        src = torch.cat((torch.tensor([[self.cls_idx 
                                       for _ in range(src.size(1))]]), src),
                        dim=0)
        # padding_mask
        pad_mask = (src == self.pad_idx).transpose(0, 1)
        
        # produce embedding
        src = self.embedding(src)
        
        # positional embedding
        pos = torch.arange(0, src.size(0)).unsqueeze(1).expand(-1, src.size(1))
        pos = self.positional_embedding(pos)
        
        # sum term to term embedding vectors
        src = torch.cat([src, pos], dim=2)
        
        attns = list()
        
        # self-attention: q, k, v = embedding
        for i, attention in enumerate(self.attentions):
            src, attn = attention(query=src, key=src, value=src,
                               key_padding_mask=pad_mask)
            attns.append(attn)
        
        # only uses CLS output
        output = src[0]
        
        return self.out(output), attns
    
    
    
    
    
class Transformer(torch.nn.Module):
    '''
    Transformer model: alternates between multihead attention, linear layers, 
    bacth normalization and residual encoding.
    
    Inputs: 
    - vocab: source vocabulary
    - hidden_dim: dimension of hidden layers
    - n_classes: number of output classes
    - num_heads: number of heads of attention
    - n_layers: number of layers
    - char2vec: Char2Vec embeddings
    - freeze: whether to freeze the char2vec embeddings
    
    Fonctions:
    - char2vec_to_tensor: matchs vocabulary tokens with char2vec embeddings 
    - forward: forward pass through the model
    '''
    def __init__(self, vocab, hidden_dim, n_classes, num_heads, n_layers,
                 char2vec=None, freeze=False):
        super().__init__()
        
        self.vocab = vocab
        self.pad_idx = self.vocab.stoi('<pad>')
        self.cls_idx = self.vocab.stoi('<cls>')
        
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.num_heads = num_heads # number of layers of attention on each token
        self.n_layers = n_layers
        
        self.char2vec = char2vec

        # sanity check
        assert self.hidden_dim % self.num_heads == 0
        
        # embeddings: pretrained or char2vec
        if char2vec is None:
            self.embedding = torch.nn.Embedding(
                num_embeddings=len(self.vocab),
                embedding_dim=self.hidden_dim,
                padding_idx=self.pad_idx)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(
                self.char2vec_to_tensor(self.vocab, 
                                        self.char2vec, 
                                        self.hidden_dim), 
                freeze=freeze, 
                padding_idx=self.pad_idx
            )
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                                         nhead=self.num_heads)
        
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=self.n_layers)
        
        self.out = torch.nn.Linear(in_features=self.hidden_dim, 
                                   out_features=self.n_classes)
        
    @staticmethod
    def char2vec_to_tensor(vocab, char2vec, hidden_dim):
        '''
        Matchs vocabulary tokens with char2vec embeddings 
        '''
        embeddings = list()
        unknown_char = list()
        for idx, char in enumerate(vocab.vocab):
            if char in char2vec.wv:
                embeddings.append(char2vec.wv[char]) 
            else:
                # sanity check
                unknown_char.append(char)
                embeddings.append(np.random.uniform(low=-1, high=1, 
                                                    size=hidden_dim))
        print("The following tokens don't have pre-trained vectors and are initialized randomly: ", 
              unknown_char)
        return torch.FloatTensor(embeddings)
        
    def forward(self, src, *args, **kwargs):
        '''
        Forward pass through the model: embedding, transformer, out.
        '''
        # concatenate token 'cls' and src
        src = torch.cat((torch.tensor([[self.cls_idx 
                                       for _ in range(src.size(1))]]), src),
                        dim=0)
        # padding_mask
        pad_mask = (src == self.pad_idx).transpose(0, 1)
        
        # produce embedding
        src = self.embedding(src)
        
        attns = list()
        
        src = self.transformer_encoder(src, src_key_padding_mask=pad_mask)
        
        # only uses CLS output
        output = src[0]
        
        return self.out(output), attns
    

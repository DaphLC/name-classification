import numpy as np
import random
import torch
import copy



class Trainer:
    '''
    The Trainer class is used to train a model. It requires a model, a learning rate,
    and a patience (maximum number of steps after the best score is obtained). It will
    keep in memory the best weights during the training phase.
    
    Criterion = Cross Entropy Loss.
    Optimizer = Adam
    
    Functions:
    - best_state_dict: returns the weights of the best performing model
    - custom_accuracy: custom accuracy score taking into account multiple outputs 
        are admissibles
    - validation: evaluate a model based on a dataloader using custom accuracy function
    - maybe_stop: early stopping
    - train: training loop
    '''
    def __init__(self, model, lr, patience=5):
        self.model = model
        self.lr = lr
        self.patience = patience
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.list_eval = list()
        self.best_weights = None
    
    @property
    def best_state_dict(self):
        '''
        Returns the weights of the best performing model based on the evaluation performed
        on test set during training phase.
        '''
        if self.best_weights is None:
            raise RuntimeError('Model has not been trained yet')
        return self.best_weights
    
    @staticmethod
    def custom_accuracy(y_true, y_pred):
        '''
        Custom accuracy score taking into account multiple outputs possible
        '''
        return np.mean([1 if pred in true else 0
                        for pred, true in zip(y_pred, y_true)])
        
    def validation(self, model, loader):
        '''
        Evaluate the model based on a dataloader using custom accuracy function
        '''
        if loader is None:
            return None
        
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

        return self.custom_accuracy(admissible_tgt, pred_tgt)

    def maybe_stop(self, score):
        '''
        Early stopping: if the score has not increased since a certain amount 
        of steps (patience), the training stops to avoid overfitting.
        The weights associated with the best model are kept in memory.
        '''
        # if no score, do no stop
        if score is None:
            return False
        
        self.list_eval.append(score)
        M = max(self.list_eval)
        M_idx = self.list_eval.index(M)
        
        # if score increases, do not stop
        if M_idx == len(self.list_eval)-1:
            self.best_weights = copy.deepcopy(self.model.state_dict())
            print("best score so far !!")
            return False
        # if score has not increased since 'patience' steps, stop
        elif M_idx <= len(self.list_eval)-1-self.patience:
            return True
        # else, do not stop
        else: 
            return False
        
    def train(self, train_loader, test_loader, epochs):
        '''
        Training loop of the model.
        The model trains on the train_loader during at least epochs steps.
        It is evaluated on the test_loader.
        '''
        counter = 0
        loss_list = []
        done = False

        for epoch in range(epochs):
            for (src, src_lengths), tgt, _ in train_loader:    
                counter += 1
                
                self.model.train()
                
                # Zero grad
                self.optimizer.zero_grad()

                # Forward pass: Compute predicted y by passing x to the model
                y_pred, attn = self.model(src, src_lengths)

                # Compute loss
                loss = self.criterion(y_pred, tgt)

                # Perform a backward pass, and update the weights.
                loss.backward()

                self.optimizer.step()
                
                loss_list.append(loss.item())
                if counter % 500 == 0:
                    # evaluate model
                    score = self.validation(self.model, test_loader)
                    print(f"Step:{counter} Loss:{sum(loss_list)/counter} Eval:{score}")
                    # stops is score does not improve
                    done = self.maybe_stop(score)
                    
                    if done: 
                        break
                        
            if done:
                break
                        
        return loss_list

    
    
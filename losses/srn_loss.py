# -*- coding: utf-8 -*-
# srn_loss.py
# author: lm


import torch
import torch.nn as nn
import torch.nn.functional as F


class SRNLoss(nn.Module):
    def __init__(self,
                 reduction='sum'):
        super(SRNLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)
        
    def forward(self, predicts, target):
        '''计算srn三个部分的loss'''
        predict = predicts['predict']
        word_predict = predicts['word_out']
        gsrm_predict = predicts['gsrm_out']
        # print(predict.size())
        # print(target.size()) 
        catsted_label = target.to(torch.int64).view(-1)
        # catsted_label = catsted_label.reshape([-1, 1])
        
        cost_word = self.loss_func(word_predict, catsted_label)
        cost_gsrm = self.loss_func(gsrm_predict, catsted_label)
        cost_vsfd = self.loss_func(predict, catsted_label)
        
        cost_word = torch.sum(cost_word)
        cost_gsrm = torch.sum(cost_gsrm)
        cost_vsfd = torch.sum(cost_vsfd)
        
        cost = cost_word * 3.0 + cost_vsfd + cost_gsrm * 0.15
        
        return {'loss': cost,
                'word_loss': cost_word,
                'img_loss': cost_vsfd}
        

if __name__ == "__main__":
    print('Test {}'.format(__file__))

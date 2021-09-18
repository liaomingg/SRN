# -*- coding: utf-8 -*-
# srn.py
# author: lm

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from model.backbone import rec_resnet_fpn
from model.head import srn_head


class SRN(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels = 38,
                 max_text_len = 25,
                 num_heads = 8,
                 num_encoders = 2,
                 num_decoders = 4,
                 hidden_dims = 512,
                 training=True):
        super(SRN, self).__init__()
        self.backbone = rec_resnet_fpn.resnet50_fpn(in_channels=in_channels)
        self.head = srn_head.SRNHead(512, # last stage's out channels
                                     out_channels,
                                     max_text_len,
                                     num_heads,
                                     num_encoders,
                                     num_decoders,
                                     hidden_dims,
                                     training)
        # self._init_weight()
        
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        
        
    def forward(self, x, others, label=None):
        x = self.backbone(x)
        predicts = self.head(x, others, label)
        return predicts
        
    

if __name__ == "__main__": 
    from data.srn_data import srn_other_inputs
    print('Test {}'.format(__file__))
    srn = SRN()
    # print(srn)
    x = torch.randn(size=(1, 1, 64, 256))
    other = srn_other_inputs([1, 64, 256], num_heads=8, max_text_len=25)
    y = srn(x, other)
    # print(y)
# -*- coding: utf-8 -*-
# srn_head.py
# author: lm


import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict

from .self_attention import WrapEncoderForFeature, WrapEncoder
import numpy as np 


class PVAM(nn.Module):
    def __init__(self,
                 in_channels,
                 char_num,
                 max_text_len,
                 num_heads,
                 num_encoders,
                 hidden_dim):
        super(PVAM, self).__init__()
        self.char_num = char_num
        self.max_len = max_text_len
        self.num_heads = num_heads
        self.num_encoders = num_encoders 
        self.hidden_dims = hidden_dim
        # Transformer encoder
        t = 256
        c = 512
        
        self.wrap_encoder_for_feature = WrapEncoderForFeature(
            src_vocab_size=1,
            max_len=t,
            n_layer=self.num_encoders,
            n_head=self.num_heads,
            d_key=self.hidden_dims // self.num_heads,
            d_value=self.hidden_dims // self.num_heads,
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepost_dropout=0.1,
            attn_dropout=0.1,
            relu_dropout=0.1)
        # PVAM
        self.flatten0 = nn.Flatten(start_dim=0, end_dim=1)
        self.fc0 = nn.Linear(in_channels, in_channels)
        self.emb = nn.Embedding(self.max_len, in_channels)
        self.flatten1 = nn.Flatten(start_dim=0, end_dim=2)
        self.fc1 = nn.Linear(in_channels, out_features=1, bias=False)
        
    def forward(self, x: torch.Tensor, encoder_word_pos, gsrm_word_pos) -> torch.Tensor:
        '''Parallel Visual Attention Module
        :param x: conv features. with shape: [1, 512, 8, 32]
        '''
        b, c, h, w = x.size() # c = 512, t = 256
        conv_features = x.reshape(shape=(b, c, h*w))     # [b, c, t]=[b, c, h*w]
        conv_features = conv_features.transpose(2, 1) # [b, t, c]
        # transformer encoder
        b, t, c = conv_features.size() # [b, t=h*w, c]
        
        enc_inputs = [conv_features, encoder_word_pos, None]
        # Transformer
        word_features = self.wrap_encoder_for_feature(enc_inputs)
        # pvam
        b, t, c = word_features.shape 
        word_features = self.fc0(word_features)
        word_features_ = word_features.reshape([-1, 1, t, c])
        word_features_ = torch.tensor(np.tile(word_features_.detach().numpy(), [1, self.max_len, 1, 1]),
                                      device = word_features_.device)
        
        word_pos_feature = self.emb(gsrm_word_pos)
        word_pos_feature_ = word_pos_feature.reshape([-1, self.max_len, 1, c])
        word_pos_feature_ = torch.tensor(np.tile(word_pos_feature_.detach().numpy(), [1, 1, t, 1]),
                                         device = word_features_.device)
        
        y = word_pos_feature_ + word_features_
        y = F.tanh(y)
        attn_weight = self.fc1(y)
        attn_weight = attn_weight.reshape([-1, self.max_len, t])
        attn_weight = F.softmax(attn_weight, dim=-1)
        pvam_features = torch.matmul(attn_weight, word_features)

        return pvam_features
    

class GSRM(nn.Module):
    def __init__(self,
                 in_channels,
                 char_num,
                 max_text_len,
                 num_heads,
                 num_encoders,
                 num_decoders,
                 hidden_dims):
        super(GSRM, self).__init__()
        self.char_num = char_num 
        self.max_len = max_text_len
        self.num_heads = num_heads 
        self.num_encoders = num_encoders 
        self.num_decoders = num_decoders 
        self.hidden_dims = hidden_dims 
        
        self.fc0 = nn.Linear(in_channels, self.char_num)
        # self.transformer0
        self.wrap_encoder0 = WrapEncoder(
            src_vocab_size=self.char_num + 1,
            max_len=self.max_len,
            n_layer=self.num_decoders,
            n_head=self.num_heads,
            d_key=self.hidden_dims // self.num_heads,
            d_value=self.hidden_dims // self.num_heads,
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepost_dropout=0.1,
            attn_dropout=0.1,
            relu_dropout=0.1)
        # self.transformer1
        self.wrap_encoder1 = WrapEncoder(
            src_vocab_size=self.char_num + 1,
            max_len=self.max_len,
            n_layer=self.num_decoders,
            n_head=self.num_heads,
            d_key=self.hidden_dims // self.num_heads,
            d_value=self.hidden_dims // self.num_heads,
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepost_dropout=0.1,
            attn_dropout=0.1,
            relu_dropout=0.1)
        ''' 
        self.mul = lambda x : torch.matmul(
            x,
            torch.transpose(self.wrap_encoder0.prepare_decoder.emb0.weight, -1, -2))
        '''
        
    def forward(self, 
                pvam_features : torch.Tensor, 
                gsrm_word_pos : torch.Tensor, 
                gsrm_self_attn_bias1=None,
                gsrm_self_attn_bias2=None):
        # GSRM Visual-to-semantic enbedding block
        b, t, c = pvam_features.size()
        pvam_features = pvam_features.reshape(shape=(-1, c))
        word_out = self.fc0(pvam_features)
        word_ids = torch.argmax(F.softmax(word_out), dim=1)
        word_ids = word_ids.reshape(shape=(-1, t, 1))
        
        # GSRM Semantic reasoning block
        pad_idx = self.char_num
        
        word1 = word_ids.to(torch.float32)
        # print('word1.size(): {}'.format(word1.size()))
        word1 = F.pad(word1, [0, 0, 1, 0], value=1.0*pad_idx)
        # print('word1.size(): {}'.format(word1.size()))
        word1 = word1.to(torch.int64)
        word1 = word1[:, :-1, :]
        # print('word1.size(): {}'.format(word1.size()))
        word2 = word_ids 
        # print('word2.size(): {}'.format(word2.size()))
        
        enc_inputs1 = [word1, gsrm_word_pos, gsrm_self_attn_bias1]
        enc_inputs2 = [word2, gsrm_word_pos, gsrm_self_attn_bias2]

        # bi-transformers
        gsrm_features1 = self.wrap_encoder0(enc_inputs1)
        gsrm_features2 = self.wrap_encoder1(enc_inputs2)
        # print('gsrm_f1.size(): {}, gsrm_f2.size(): {}'.format(gsrm_features1.size(), gsrm_features2.size()))
        
        gsrm_features2 = F.pad(gsrm_features2, [0, 0, 0, 1], value=0.)
        # print('gsrm_f2.size(): {}'.format(gsrm_features2.size()))
        gsrm_features2 = gsrm_features2[:, 1:, :]
        
        gsrm_features = gsrm_features1 + gsrm_features2
        
        # print("emb0.weight.size(): {}".format(self.wrap_encoder0.prepare_decoder.emb0.weight.size()))
        gsrm_out = torch.matmul(
            gsrm_features,
            torch.transpose(self.wrap_encoder0.prepare_decoder.emb0.weight, -1, 2))
        
        # gsrm_out = self.mul(gsrm_features)
        
        b, t, c = gsrm_out.shape 
        gsrm_out = gsrm_out.reshape([-1, c])
        return gsrm_features, word_out, gsrm_out 
        
        
class VSFD(nn.Module):
    ''''''
    def __init__(self,
                 in_channels=512,
                 pvam_channels=512,
                 char_num=38):
        super(VSFD, self).__init__()
        self.char_num = char_num 
        self.fc0 = nn.Linear(in_channels * 2, pvam_channels)
        self.fc1 = nn.Linear(pvam_channels, self.char_num)
        
    def forward(self, pvam_features: torch.Tensor, gsrm_features: torch.Tensor):
        ''''''
        b, t, c1 = pvam_features.size() # c1 = 512
        b, t, c2 = gsrm_features.size() # c2 = 512
        features = torch.cat([pvam_features, gsrm_features], dim=2) # [b, t, c1+c2]
        features = features.reshape(shape=(-1, c1+c2))
        features = self.fc0(features) # [b, t, c1]
        features = torch.sigmoid(features)
        features = features.reshape(shape=(-1, t, c1)) # [b, t, c1]
        
        features = features * pvam_features + (
            1.0 - features) * gsrm_features
        
        features = features.reshape(shape=(-1, c1))
        
        out = self.fc1(features) # [b*t, char_num]
        
        return out 
    
    
        
        
        
        


class SRNHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 max_text_len,
                 num_heads,
                 num_encoders,
                 num_decoders,
                 hidden_dims,
                 training=True):
        super(SRNHead, self).__init__()
        self.char_num = out_channels # 字符数量（包含结束符)
        self.max_len = max_text_len
        self.num_heads = num_heads
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders 
        self.hidden_dims = hidden_dims
        self.training = training
        
        self.pvam = PVAM(in_channels, 
                         self.char_num, 
                         max_text_len, 
                         num_heads, 
                         num_encoders, 
                         hidden_dims)
        
        self.gsrm = GSRM(
            in_channels = in_channels,
            char_num = self.char_num,
            max_text_len = self.max_len,
            num_heads = self.num_heads,
            num_encoders = self.num_encoders,
            num_decoders = self.num_decoders,
            hidden_dims = self.hidden_dims
        )
        
        self.vsfd = VSFD(in_channels, char_num=self.char_num)
        
        self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.gsrm.wrap_encoder0.prepare_decoder.emb0
        
        
    def forward(self, x, data, targets=None):
        '''
        :param x: input feature.
        :param others: other_inputs, such as enc_word_pos.
        :param targets: the targets of x.
        '''
        encoder_word_pos = data['encoder_word_pos']
        gsrm_word_pos = data['gsrm_word_pos']
        gsrm_self_attn_bias1 = data['gsrm_self_attn_bias1']
        gsrm_self_attn_bias2 = data['gsrm_self_attn_bias2']
        
        pvam_features = self.pvam(x, encoder_word_pos, gsrm_word_pos)
        
        gsrm_features, word_out, gsrm_out = self.gsrm(
            pvam_features,
            gsrm_word_pos,
            gsrm_self_attn_bias1,
            gsrm_self_attn_bias2)
        
        final_out = self.vsfd(pvam_features, gsrm_features)
        
        if not self.training:
            final_out = F.softmax(final_out, dim=1)
        
        _, decode_out = torch.topk(final_out, k=1)
        
        predicts = OrderedDict([
            ('predict', final_out),
            ('pvam_feature', pvam_features),
            ('decode_out', decode_out),
            ('word_out', word_out),
            ('gsrm_out', gsrm_out)
        ])
        
        return predicts
    

if __name__ == "__file__":
    print('Test {}'.format(__file__))
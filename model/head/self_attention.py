# -*- coding: utf-8 -*-
# self_attention.py
# author: lm

'''
paper: Attention is all you need.
arXiv:1706.03762v5
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self._init_weight()
    
    def _init_weight(self):
        nn.init.constant_(self.norm.weight, 1.)
        nn.init.constant_(self.norm.bias, 0.)
        
    def forward(self, x, **kwrags):
        return self.fn(self.norm(x), **kwrags)


class PostDropout(nn.Module):
    def __init__(self,
                 fn,
                 dropout=0.):
        super(PostDropout, self).__init__()
        self.dropout = dropout
        self.fn = fn
    
    def forward(self, x):
        return F.dropout(self.fn(x), self.dropout)
    
    
class AddResidual(nn.Module):
    def __init__(self,
                 fn):
        super(AddResidual, self).__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)
    
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
        
    def forward(self, x):
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head
        
        self.heads = heads 
        self.scale = dim_head ** -0.5 
        
        self.attend = nn.Softmax(dim = -1)
        # q, k, v in one linear
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if (heads == 1 and dim == dim_head) else nn.Identity()
        
    def forward(self, x):
        b, t, c, h = *x.shape, self.heads
        # print('batch_size, max_text_len, channels, heads')
        # print(b, t, c, h)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1) # [b, t, h*c]
        # print(q.size(), k.size(), v.size())
        q = q.reshape(b, t, h, -1).permute(0, 2, 1, 3) # [b, h, t, c]
        k = k.reshape(b, t, h, -1).permute(0, 2, 1, 3).transpose(2, 3) # [b, h, c, t]
        v = v.reshape(b, t, h, -1).permute(0, 2, 1, 3) # [b, h, t, c]
        w = self.attend(self.scale * torch.matmul(q, k)) # [b, h, t, t]
        
        out = torch.matmul(w, v) # [b, h, t, c]
        out = out.permute(0, 2, 1, 3).reshape(b, t, -1)
        out = self.to_out(out)
        return out 
        
        
class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.):
        super(Transformer, self).__init__()      
        self.layers = nn.ModuleList([]) 
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
                ]))
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x 
        return x 


class Encoder(nn.Module):
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value, 
                 d_model,
                 d_inner_hid,
                 prepost_drouput,
                 attn_dropout,
                 relu_dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                nn.ModuleList([
                    AddResidual(PostDropout(PreNorm(d_model, Attention(d_model, n_head, d_key, attn_dropout)), prepost_drouput)),
                    AddResidual(PostDropout(PreNorm(d_model, FeedForward(d_model, d_inner_hid, relu_dropout)), prepost_drouput))
                ]))
        self.norm = nn.LayerNorm(d_model) 
        nn.init.constant_(self.norm.weight, 1.)
        nn.init.constant_(self.norm.bias, 0.)
       
    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x)
            x = feedforward(x)
        x = self.norm(x)
        return x 
    

class WrapEncoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 max_len,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepost_dropout,
                 attn_dropout,
                 relu_dropout,
                 bos_idx=0
                 ):
        super().__init__()
        self.prepare_decoder = PrepareDecoder(src_vocab_size, 
                                              d_model, 
                                              max_len, 
                                              prepost_dropout, 
                                              bos_idx)
        self.encoder = Encoder(n_layer,
                               n_head,
                               d_key,
                               d_value,
                               d_model,
                               d_inner_hid,
                               prepost_dropout,
                               attn_dropout,
                               relu_dropout)
        
    def forward(self, enc_inputs):
        # assure attn_bias is None
        src_word, src_pos, src_slf_attn_bias = enc_inputs
        # ('src_word.size(), src_pos.size() in WrapEncoder forward.')
        # print(src_word.size(), src_pos.size()) 
        enc_input = self.prepare_decoder(src_word, src_pos)
        enc_output = self.encoder(enc_input)
        return enc_output
    

class WrapEncoderForFeature(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 max_len,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepost_dropout,
                 attn_dropout,
                 relu_dropout,
                 bos_idx=0):
        super(WrapEncoderForFeature, self).__init__()
        self.prepare_encoder = PrepareEncoder(
            src_vocab_size,
            d_model,
            max_len,
            prepost_dropout,
            bos_idx)
        self.encoder = Encoder(n_layer,
                               n_head,
                               d_key,
                               d_value,
                               d_model,
                               d_inner_hid,
                               prepost_dropout,
                               attn_dropout,
                               relu_dropout)
    
    def forward(self, enc_inputs):
        conv_features, src_pos, src_self_attn_bias = enc_inputs
        enc_inputs = self.prepare_encoder(conv_features, src_pos)
        enc_out  = self.encoder(enc_inputs)
        return enc_out 


class PrepareEncoder(nn.Module):
    '''先对由CNN提取的特征进行预处理'''
    def __init__(self,
                 src_vocab_size, # char_number
                 src_emb_dim,
                 src_max_len,
                 dropout = 0.0,
                 bos_idx = 0,
                 word_emb_param_name = None,
                 pos_enc_param_name = None):
        super(PrepareEncoder, self).__init__()   
        self.src_emb_dim = src_emb_dim
        self.src_max_len = src_max_len
        self.scale = self.src_emb_dim ** 0.5
        self.emb = nn.Embedding(
            num_embeddings=self.src_max_len,
            embedding_dim=self.src_emb_dim)
        self.dropout = dropout
        
    def forward(self, src_word : torch.Tensor, src_pos : torch.Tensor):
        '''
        :param src_word:
        :param src_pos: 字符的位置编码信息
        '''
        src_word_emb = src_word 
        src_word_emb = src_word_emb.to(torch.float32)
        src_word_emb *= self.scale 
        
        src_pos = torch.squeeze(src_pos, dim=-1)
        src_pos_enc = self.emb(src_pos)
        # src_pos_enc.requires_grad = False 
        src_pos_enc_no_grad = src_pos_enc.detach()
        enc_input = src_word_emb + src_pos_enc_no_grad
        if self.dropout:
            enc_input = F.dropout(enc_input, p=self.dropout)
        return enc_input
        

class PrepareDecoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout=0.,
                 bos_idx=0,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super().__init__()
        self.src_emb_dim = src_emb_dim
        # print('src_vocab_size: {}'.format(src_vocab_size))
        self.emb0 = nn.Embedding(src_vocab_size,
                                 src_emb_dim,
                                 padding_idx=bos_idx)
        self.emb1 = nn.Embedding(src_max_len,
                                 src_emb_dim)
        self.dropout = dropout
        self.scale = src_emb_dim ** 0.5
        self._init_weight()
    
    def _init_weight(self):
        nn.init.normal_(self.emb0.weight, 0., std=self.src_emb_dim**-0.5)
        
    def forward(self, src_word : torch.Tensor, src_pos : torch.Tensor):
        # print('src_word.size(), src_pos.size()')
        # print(src_word.size(), src_pos.size())
        src_word = src_word.to(torch.int64)
        src_word = torch.squeeze(src_word, dim=-1)
        src_word_emb = self.emb0(src_word)
        src_word_emb = src_word_emb * self.scale
        
        src_pos = torch.squeeze(src_pos, dim=-1)
        src_pos_enc = self.emb1(src_pos)
        # src_pos_enc.requires_grad = False
        src_pos_enc_no_grad = src_pos_enc.detach()
        # print("src_word_emb.size(), src_pos_enc_no_grad.size()")
        # print(src_word_emb.size(), src_pos_enc_no_grad.size())
        enc_input =  src_word_emb + src_pos_enc_no_grad
        if self.dropout:
            enc_input = F.dropout(enc_input, self.dropout)
            
        return enc_input
        
        

class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention
    '''
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head = 1,
                 dropout_rate = 0.):
        super(MultiHeadAttention, self).__init__()
        self.d_key = d_key # assume d_query == d_key
        self.d_value = d_value
        self.d_model = d_model
        self.scale = self.d_model ** -0.5
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        
        self.q_fc = nn.Linear(d_model, n_head*d_key, bias=False)
        self.k_fc = nn.Linear(d_model, n_head*d_key, bias=False)
        self.v_fc = nn.Linear(d_model, n_head*d_value, bias=False)
        self.p_fc = nn.Linear(n_head*d_value, d_model, bias=False)
        
    def _prepare_qkv(self, queries, keys, values):
        if keys is None:
            keys, values = queries, queries
        
        q = self.q_fc(queries) # [N, T, n_head*dkey]
        b, t, _ = q.size()
        q = torch.reshape(q, shape=(b, t, self.n_head, self.d_key))
        q = q.permute(dims=(0, 2, 1, 3))
        
        k = self.k_fc(keys)
        k = torch.reshape(k, shape=(b, t, self.n_head, self.d_key))
        k = k.permute(dims=(0, 2, 1, 3)) # [N, n_head, t, d_key]
        
        v = self.v_fc(values)
        v = torch.reshape(v, shape=(b, t, self.n_head, self.d_value))
        v = v.permute(dims=(0, 2, 1, 3))
        
        return q, k, v     
    
    def forward(self, queries, keys, values, attn_bias=None):
        '''Do Scaled Dot-Product Attention in Multi-Head Attention.
        :param queries:
        :param keys:
        :param values:
        :attn_bias: 
        '''
        keys = queries if keys is None else keys
        values = keys if values is None else values 
        
        q, k, v = self._prepare_qkv(queries, keys, values)
        
        w = torch.matmul(q, torch.transpose(k, 2, 3))
        w *= self.scale 
        w = F.softmax(w)
        if self.dropout_rate:
            w = F.dropout2d(w, p = self.dropout_rate)
            
        out = torch.matmul(w, v)
        
        out = out.permute(dims=(0, 2, 1, 3))
        b, t, n, d = out.size()
        out = torch.reshape(out, shape=(b, t, n*d))
        
        out = self.p_fc(out)
        
        return out 
    

class PrePostProcessLayer(nn.Module):
    '''PrePostProcessLayer'''
    def __init__(self,
                 process_cmd,
                 d_model,
                 dropout):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd 
        self.functors = []
        for cmd in self.process_cmd:
            if cmd == 'a':
                self.functors.append(lambda x, y: x + y if y is not None else x)
            elif cmd == 'n':
                self.functors.append(
                    nn.LayerNorm(d_model))
            elif cmd == 'd':
                self.functors.append(lambda x: F.dropout(x, dropout) if dropout else x)
    
    def forward(self, x, y):
        for i, fn in enumerate(self.functors):
            if fn == 'a':
                x = self.functors[i](x, y)
            else:
                x = self.functors[i](x)
        return x 
    

class EncoderLayer(nn.Module):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 pre_cmd = 'n',
                 post_cmd = 'da'):
        super(EncoderLayer, self).__init__()
        # pre is layer normal
        self.pre1 = PrePostProcessLayer(pre_cmd, d_model, prepostprocess_dropout)
        self.attn = MultiHeadAttention(d_key, d_value, d_model, n_head, attention_dropout)
        self.post1 = PrePostProcessLayer(post_cmd, d_model, prepostprocess_dropout)
        
        self.pre2 = PrePostProcessLayer(pre_cmd, d_model, prepostprocess_dropout)
        self.ffn = FeedForward(d_model, d_inner_hid, relu_dropout)
        self.post2 = PrePostProcessLayer(post_cmd, d_model, prepostprocess_dropout)
        
    def forward(self, enc_input, attn_bias):
        attn_output = self.attn(self.pre1(enc_input), None, None)
        attn_output = self.post1(attn_output, enc_input)
        
        ffn_out = self.ffn(self.pre1(attn_output))
        ffn_out = self.post2(ffn_out, attn_output)
        

if __name__ == "__main__":
    
    print('Test {}'.format(__file__))

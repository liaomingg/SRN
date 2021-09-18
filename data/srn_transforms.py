# -*- coding: utf-8 -*-
# srn_transform.py
# author: lm

import torch
from torchvision import transforms
import numpy as np 
from PIL import Image 

from .srn_data import process_image_srn, resize_normalize_srn, srn_other_inputs


class SRNResizeImg(object):
    def __init__(self,
                 image_shape=[1, 64, 256],
                 num_heads=8,
                 max_text_len=25) -> None:
        super(SRNResizeImg, self).__init__()
        self.image_shape = image_shape
        self.num_heads = num_heads
        self.max_text_len = max_text_len
        
    def __call__(self, data) -> dict:
        '''
        :param data: a dict.
        '''
        data['image'] = resize_normalize_srn(data['image'], self.image_shape)
        [encoder_word_pos, 
         gsrm_word_pos, 
         gsrm_self_attn_bias1, 
         gsrm_self_attn_bias2] = srn_other_inputs(self.image_shape, 
                                                  self.num_heads, 
                                                  self.max_text_len)
        data['encoder_word_pos'] = encoder_word_pos
        data['gsrm_word_pos'] = gsrm_word_pos
        data['gsrm_self_attn_bias1'] = gsrm_self_attn_bias1
        data['gsrm_self_attn_bias2'] = gsrm_self_attn_bias2
        return data 


class SRNCollateFN(object):
    def __init__(self) -> None:
        super(SRNCollateFN, self).__init__()

    def __call__(self, batch):
        '''
        collate function for srn data.
        :param batch: data batch, keys in data:
            image:
            label:
            length:
            encoder_word_pos:
            gsrm_word_pos:
            gsrm_self_attn_bias1:
            gsrm_self_attn_bias2:
        ''' 
        # 将srn输入数据组装成batch
        to_tensor_keys = []
        new_data = {}
        for data in batch:
            for k, v in data.items():
                if k not in new_data.keys():
                    new_data[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, Image.Image)):
                # if isinstance(v, torch.Tensor):
                    # print(k, type(v), v.dtype)
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                    v = torch.Tensor(v)
                    new_data[k].append(torch.Tensor(v))
                else:
                    new_data[k].append(v) # list.append
        for k in to_tensor_keys:
            new_data[k] = torch.stack(new_data[k], dim=0)
            if 'pos' in k or 'label' in k:
                new_data[k] = new_data[k].to(torch.int64)
        return new_data
    

if __name__ == "__main__":
    print('Test {}'.format(__file__))
    
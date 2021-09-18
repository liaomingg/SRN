# -*- coding: utf-8 -*-
# srn_data.py modified from ppocr
# author: lm

import cv2 as cv
import numpy as np
import torch


def resize_normalize_srn(cv_img, shape = [1, 64, 256]):
    '''对于SRN模型，需要将输入的图像，根据shape进行resize padding'''
    C, H, W = shape 
    h, w = cv_img.shape[:2]
    
    new_img = np.zeros((H, W))
    
    W = min(W, H * ((w - 1) // h + 1))
    img = cv.resize(cv_img, (W, H))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    new_img[:, :img.shape[1]] = img
    new_img = new_img[:, :, np.newaxis] # expand dim 
    H, W, C = new_img.shape # C == 1
    return np.reshape(new_img, (C, H, W)).astype(np.float32)


def srn_other_inputs(shape, num_heads=8, max_text_len=25, stride=8, eps=-1e9):
    '''SRN 模型的其他输入, 单个图片的输入，如果是inference，需要添加一个维度组成batch'''
    C, H, W = shape
    feature_dim = int((H / stride) * (W / stride))
    
    encoder_word_pos = np.array(range(0, feature_dim)).reshape(
        (feature_dim, 1)).astype(np.int64)
    gsrm_word_pos = np.array(range(0, max_text_len)).reshape(
        (max_text_len, 1)).astype(np.int64)
    
    gsrm_attn_bias_data = np.ones((1, max_text_len, max_text_len))
    # 上三角矩阵
    gsrm_self_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
        [1, max_text_len, max_text_len])
    gsrm_self_attn_bias1 = np.tile(
        gsrm_self_attn_bias1,
        [num_heads, 1, 1]).astype('float32') * [eps]
    
    # 下三角矩阵
    gsrm_self_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
        [1, max_text_len, max_text_len])
    gsrm_self_attn_bias2 = np.tile(
        gsrm_self_attn_bias2,
        [num_heads, 1, 1]).astype('float32') * [eps]
        
    # encoder_word_pos = encoder_word_pos[np.newaxis, :]
    # gsrm_word_pos = gsrm_word_pos[np.newaxis, :]
    '''  
    return [torch.Tensor(encoder_word_pos).to(torch.int64), 
            torch.Tensor(gsrm_word_pos).to(torch.int64), 
            torch.Tensor(gsrm_self_attn_bias1),
            torch.Tensor(gsrm_self_attn_bias2)]   
    '''     
    return [encoder_word_pos,
            gsrm_word_pos,
            gsrm_self_attn_bias1,
            gsrm_self_attn_bias2]  
    
def process_image_srn(cv_img, shape, num_heads, max_text_len):
    '''处理输入给SRN模型的图像数据'''
    norm_img = resize_normalize_srn(cv_img, shape)
    norm_img = norm_img[np.newaxis, :] # [1, 1, H, W]
    
    [encoder_word_pos, \
     gsrm_word_pos, \
     gsrm_self_attn_bias1, \
     gsrm_self_attn_bias2] = srn_other_inputs(shape, num_heads, max_text_len)

    # expand dims
    gsrm_self_attn_bias1 = gsrm_self_attn_bias1.astype(np.float32)
    gsrm_self_attn_bias2 = gsrm_self_attn_bias2.astype(np.float32)
    encoder_word_pos = encoder_word_pos.astype(np.int64)
    gsrm_word_pos = gsrm_word_pos.astype(np.int64)
    # there are all numpy data. convert to torch.Tensor 
    return (norm_img,
            encoder_word_pos,
            gsrm_word_pos,
            gsrm_self_attn_bias1,
            gsrm_self_attn_bias2)
        
        
if __name__ == "__main__":
    import os 
    path = './imgs_words_en/word_401.png'
    if not os.path.exists(path):
        print('{} not exist.'.format(path))
        exit(0)
        
    cv_img = cv.imread(path)
    print("input cv_img.shape:", cv_img.shape)
    items = process_image_srn(cv_img, shape=[1, 64, 256], num_heads=8, max_text_len=25)
    norm_img, encoder_word_pos, gsrm_word_pos, gsrm_self_attn_bias1, gsrm_self_attn_bias2 = items
    
    for item in items:
        print()
        print(item)
    
    

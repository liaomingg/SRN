# -*- coding: utf-8 -*-
# lmdb_dataset.py
# author: lm


import os

import cv2
import lmdb
import numpy as np
from torch.utils.data import Dataset


class LMDBDataSet(Dataset):
    def __init__(self, data_dir, shuffle=True, transform = None, target_transform = None, seed=None):
        super(LMDBDataSet, self).__init__()

        self.do_shuffle = True
        self.transform = transform 
        self.target_transform = target_transform 

        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        print("Initialize indexs of datasets:%s" % data_dir)
        self.data_idx_order_list = self.dataset_traversal()
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath": dirpath, 
                                          "env":env,
                                          "txn":txn, 
                                          "num_samples":num_samples}
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'],
                                                file_idx)
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        data = {'image': self.get_img_data(img), 
                'label': label}
        if self.transform:
            # transform input image.
            data = self.transform(data)
        
        if self.target_transform:
            # transform input label.
            data = self.target_transform(data)
        
        if data is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return data

    def __len__(self):
        return self.data_idx_order_list.shape[0]


if __name__ == "__main__":
    from srn_transforms import SRNResizeImg
    from alphabet import Alphabet
    import string 
    import torch
    lmdb_path = '/Users/liaoming/Downloads/SRN'
    image_shape = [1, 64, 256]
    image_transform = SRNResizeImg(image_shape, num_heads=8, max_text_len=25)
    alphabet = Alphabet(string.digits + string.ascii_lowercase)
    lmdb_dataset = LMDBDataSet(lmdb_path, 
                               transform = image_transform,
                               target_transform=alphabet.transform_label)
    print(len(lmdb_dataset))
    data = lmdb_dataset[0]
    print(data.keys())
    '''
    image (1, 64, 256)
    label (25,)
    length ()
    encoder_word_pos (256, 1)
    gsrm_word_pos (25, 1)
    gsrm_slf_attn_bias1 (8, 25, 25)
    gsrm_slf_attn_bias2 (8, 25, 25)
    '''
    for k in data.keys():
        print(k)
        if isinstance(data[k], torch.Tensor):
            print(data[k].size())
        elif isinstance(data[k], np.ndarray):
            print(data[k].shape)
    # print(data['image'])
            
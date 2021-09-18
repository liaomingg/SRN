# -*- coding: utf-8 -*-
# alphabet.ppy
# author: lm

import string
import numpy as np

class Alphabet(object):
    def __init__(self,
                 alphabet = '',
                 max_len = 25,
                 beg = 'SOS',
                 end = 'EOS') -> None:
        super(Alphabet, self).__init__()
        self.max_len = max_len 
        self.beg = beg 
        self.end = end 
        self.alphabet = self._add_beg_end(alphabet)
        self.char_num = len(self.alphabet)
        self.dict = {}
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx 
        print(self.dict)
            
            
    def _add_beg_end(self, alphabet):
        return list(alphabet) + [self.beg, self.end]
    
    
    def encode(self, text_label, ignore_others=True):
        index_label = []
        for char in text_label:
            if char not in self.dict.keys() and ignore_others:
                print('char: `{}` in text_label: `{}` not supported by alphabet.'.format(char, text_label))
                continue 
            index_label.append(self.dict[char])
        
        return index_label
    
    
    def decode(self, index_label, remove_duplicated=False):
        '''srn not remove duplicated. ctc should remove duplicated.'''
        chars = []
        for idx, index in enumerate(index_label):
            index = int(index)
            if remove_duplicated:
                if idx > 0 and index_label[idx - 1] == index_label[idx]:
                    continue
            if self.alphabet[index] == self.end:
                break
            chars.append(self.alphabet[index])
        
        return ''.join(chars)
    
    def decode_batch(self, index_labels, remove_duplicate=True):
        '''批量解码'''
        return [self.decode(index_label, remove_duplicate) for index_label in index_labels]
    
    def transform_label(self, data):
        text_label = data['label']
        data['length'] = len(data['label'])
        index_label = np.array(self.encode(text_label), dtype=np.int64)
        index_label_padded = np.ones(shape=self.max_len, dtype=np.int64) * (self.char_num - 1)
        index_label_padded[:len(index_label)] = index_label
        # index_label = index_label_padded.reshape(-1, 1)
        data['label'] = index_label_padded
        return data
    

    
    
    
if __name__ == "__main__":
    '''
    text: hackishes
    encoded text: [17, 10, 12, 20, 18, 28, 17, 14, 28]
    train label: [17 10 12 20 18 28 17 14 28 37 37 37 37 37 37 37 37 37 37 37 37 37 37 37 37]
    '''
    data = {}
    data['label'] = 'hackishsess'
    alphabet = Alphabet(string.digits + string.ascii_lowercase, max_len=25)
    index_label_padded = alphabet.transform_label(data)
    print('index_label_padded:', index_label_padded)
    decoded_label = alphabet.decode(index_label_padded['label'])
    print(decoded_label)
    
# -*- coding: utf-8 -*-
# rec_metric.py
# author: lm

import Levenshtein

class RecMetric(object):
    def __init__(self,
                 main_indicator='acc',
                 ) -> None:
        super().__init__()
        self.main_indicator = main_indicator
        self.reset()
        
    def reset(self):
        self.correct_num = 0 
        self.total_num = 0
        self.norm_edit_dist = 0
        
    def get_metric(self):
        acc = 1.0 * self.correct_num / self.total_num
        norm_edit_dist = 1 - self.norm_edit_dist / self.total_num
        self.reset()
        return {'acc': acc,
                'norm_edit_dist': norm_edit_dist}
        
    def __call__(self, preds, labels):
        correct_num = 0
        total_num = 0
        norm_edit_dist = 0.0
        # print("Preds:", preds, "Labels:", labels)
        print([each for each  in zip(preds, labels)])
        for pred, label in zip(preds, labels):
            pred = pred.replace(' ', '')
            label = label.replace(' ', '')
            norm_edit_dist += Levenshtein.distance(pred, label) / (max(len(pred), len(label), 1))
            if pred == label:
                correct_num += 1
            total_num += 1
            
        self.correct_num += correct_num 
        self.total_num += total_num
        self.norm_edit_dist += norm_edit_dist
        return {'acc': correct_num / total_num,
                'norm_edit_dist': 1 - norm_edit_dist / self.total_num}
            
# -*- coding: utf-8 -*-
# utils.py
# author: lm
import torch 
import shutil

class AverageMeter(object):
    def __init__(self,
                 name = '',
                 fmt = ':f') -> None:
        super(AverageMeter, self).__init__()
        self.name = name 
        self.fmt = fmt 
        self.reset()
        
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0 
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n 
        self.count += n 
        if self.count:
            self.avg = self.sum / self.count 
            
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def __call__(self, val, n=1):
        self.update(val, n)
        

class ProgressMeter(object):
    def __init__(self,
                 num_batches,
                 meters,
                 prefix='') -> None:
        super(ProgressMeter, self).__init__()
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters 
        self.prefix = prefix 
        
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
        

def load_checkpoint(model, ckpt_path):
    # load to cpu, may trained on other gpu id.
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    state = checkpoint 
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            ckpt[k[7:]] = v # remove the prefix module. of parallel model.
        else:
            ckpt[k] = v
    model.load_state_dict(ckpt)
    return state 


def save_checkpoint(state, ckpt_path, is_best=False, best_ckpt_path=None):
    torch.save(state, ckpt_path)
    if is_best and best_ckpt_path:
        print(' ---> find best model: {}'.format(best_ckpt_path))
        shutil.copyfile(ckpt_path, best_ckpt_path)
    


if __name__ == "__main__":
    print('Test {}'.format(__file__))
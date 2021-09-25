# -*- coding: utf-8 -*-
# train.py
# author: lm


import argparse
import json
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import yaml
from easydict import EasyDict as edict
from torch import optim
from torch.utils.data.dataloader import DataLoader

from data.alphabet import Alphabet
from data.lmdb_dataset import LMDBDataSet
from data.srn_transforms import SRNCollateFN, SRNResizeImg
from losses.srn_loss import SRNLoss
from metrics.rec_metric import RecMetric
from scripts.lr_schedule import WarmupCosineAnnealingLR
from scripts.utils import (AverageMeter, ProgressMeter, load_checkpoint,
                           save_checkpoint)
from srn import SRN

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

def sgd_optimizer(model:nn.Module, lr:float, momentum:float, weight_decay:float):
    params = []
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr 
        if 'bias' in k or 'bn' in k:
            apply_weight_decay = 0
            print('set weight decay = 0 for {}'.format(k))
        if 'bias' in k:
            apply_lr = 2 * lr # caffe 
        params += [{'params': [v], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer




def main(args):
    # check cpu or gpu!
    if not args.gpu and torch.cuda.is_available():
        args.gpu = [int(gpu) for gpu in args.gpu.split(',')]
        print('Using GPU: {} for training.'.format(args.gpu))
    else:
        args.gpu = None 
        print('Using CPU for training.')
    
    # distributed training
    if args.distributed:
        # default init_method is 'env://'
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = os.environ['RANK']
            
        if args.multi_processing_distributed:
            args.rank = args.rank * args.gpus_per_node + args.gpu
            
        distributed.init_process_group(backend = args.dist_backend,
                                       init_method = args.dist_url,
                                       world_size = args.world_size,
                                       rank = args.rank)
    
    alphabet = Alphabet(args.alphabet, args.head.max_text_len)
    train_dataset, train_loader = [], [] 
    if args.dataset.train.enable:
        print('Loading train dataset...')
        train_dataset = LMDBDataSet(args.dataset.train.path,
                                    args.dataset.train.shuffle,
                                    SRNResizeImg(args.dataset.train.image_shape,
                                                args.head.num_heads,
                                                args.head.max_text_len),
                                    target_transform=alphabet.transform_label)
        train_sampler = None 
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                batch_size=args.dataset.train.batch_size,
                                shuffle=args.dataset.train.shuffle,
                                num_workers=args.workers,
                                pin_memory=True,
                                sampler=train_sampler,
                                collate_fn=SRNCollateFN())
    else:
        print('Skip training process.')
        
    # load validate dataset.
    val_dataset, val_loader = [], [] 
    if args.dataset.val.enable:
        print('Loading validate dataset...')
        val_dataset = LMDBDataSet(args.dataset.val.path,
                                args.dataset.val.shuffle,
                                SRNResizeImg(args.dataset.val.image_shape,
                                            args.head.num_heads,
                                            args.head.max_text_len),
                                target_transform=alphabet.transform_label)
        
        val_loader = DataLoader(val_dataset,
                                batch_size=args.dataset.val.batch_size,
                                shuffle=args.dataset.val.shuffle,
                                num_workers=args.workers,
                                pin_memory=True,
                                collate_fn=SRNCollateFN())
    else:
        print('Skip validate during training process.')
    
    print('Train examples: {}, Val examples: {}.'.format(len(train_dataset), len(val_dataset)))
    
    model = SRN(args.model.in_channels,
              alphabet.char_num,
              args.head.max_text_len,
              args.head.num_heads,
              args.head.num_encoders,
              args.head.num_decoders,
              args.head.hidden_dims,
              training=True)
    
    if args.model.display:
        print('Model structure.')
        print(model)


    best_metric = {
        'acc': 0.0,
        'norm_edit_dist': 0.0}
    if args.resume:
        # maybe load checkpoint before move to cuda.
        if os.path.isfile(args.resume):
            print(' ===> resume parameters from: {}'.format(args.resume))
            state = load_checkpoint(model, args.resume)
            best_metric = state['metrics']
            print('Load best metric:', best_metric)
        else:
            print(' xxx> no checkpoint found at: {}'.format(args.resume))
     

    # set GPU or CPU
    if args.gpu is None or not torch.cuda.is_available():
        print('CUDA is unavailable!!! Using CPU training will be slow!!!')
          
    elif args.distributed:
            model = model.cuda()   
            model = torch.nn.parallel.DistributedDataParallel(model, 
                                                              find_unused_parameters=True)
            print('DistributedDataparallel training with selected GPUS: {}'.format(args.gpu))
            
    elif len(args.gpu) <= 1:
        model = model.cuda()
        print('Training model with single GPU: {}'.format(args.gpu))
        
    else:
        model = torch.nn.DataParallel(model).cuda()
        print('Training model with Data Parallel.')
        
    # define criterion
    criterion = SRNLoss('mean')
    if args.gpu:
        criterion = criterion.cuda()
        
    # optimizer
    # optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay) 
    optimizer = optim.Adam(model.parameters(), args.lr, amsgrad=True)
    # lr_schedule
    lr_scheduler = WarmupCosineAnnealingLR(optimizer, 
                                           args.epochs * len(train_loader) // args.gpus_per_node, 
                                           warmup=args.warmup)
    if args.cudnn_benchmark:
        cudnn.benchmark = True 

    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler:
                train_sampler.set_epoch(epoch)
                
        train(args, model, train_loader, criterion, optimizer, lr_scheduler, alphabet, epoch)
        
        if args.dataset.val.enable and epoch % args.dataset.val.interval == 0:
            metric = val(args, model, val_loader, criterion, alphabet, epoch)
        
        if not args.multi_processing_distributed or (args.multi_processing_distributed and args.rank % args.gpus_per_node == 0):
            state = {
                'epoch': epoch,
                'name': args.name,
                'state_dict': model.state_dict(),
                'metrics': metric,
                'optimizer': optimizer,
                'scheduler': lr_scheduler.state_dict()}
            save_checkpoint_srn(state, args, best_metric['acc'] <= metric['acc'])
            if best_metric['acc'] <= metric['acc']:
                best_metric = metric
                
    print('Train complete!')


    
def train(args : edict, 
          model : nn.Module, 
          data_loader : DataLoader,
          criterion : SRNLoss,
          optimizer : optim.Optimizer, 
          lr_scheduler, 
          alphabet: Alphabet,
          epoch: int):
    ''''''
    batch_time = AverageMeter('BatchTime', ':.3f')
    data_time = AverageMeter('DataTime', ':.3f')
    loss = AverageMeter('Loss', ':.3f')
    img_loss = AverageMeter('ImageLoss', ':.3f')
    word_loss = AverageMeter('WordLoss', ':.3f')
    acc = AverageMeter('Acc', ':.3f')
    norm_edit_dist = AverageMeter('NormEditDist', ':.3f')
    progress = ProgressMeter(len(data_loader),
        [batch_time, data_time, loss, img_loss, word_loss, acc, norm_edit_dist],
        prefix='Epoch: {}'.format(epoch))
    
    metric = RecMetric()
    
    print('Epoch: {}, lr: {:.3e}'.format(epoch, lr_scheduler.get_lr()[0]))
    model.train() # convert to training mode.
    
    st = time.time()
    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - st)
        # to cuda
        if args.gpu:
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
                    
        preds = model(data['image'], data, data['label'])
        losses = criterion(preds, data['label'])
        
        '''
        {'loss': cost,
         'word_loss': cost_word,
         'img_loss': cost_vsfd}
        '''
        model.zero_grad()
        losses['loss'].backward()
        # gradient clip
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()   
        
        loss.update(losses['loss'].item())
        img_loss.update(losses['img_loss'].item())
        word_loss.update(losses['word_loss'].item())
        
        batch_indexs = torch.argmax(preds['predict'], dim=-1).reshape(-1, alphabet.max_len)
        
        m = metric(alphabet.decode_batch(batch_indexs), 
                   alphabet.decode_batch(data['label']))
        
        acc.update(m['acc'])
        norm_edit_dist.update(m['norm_edit_dist'])
        
        batch_time.update(time.time() - st)     
        st = time.time()
        lr_scheduler.step()
        if idx % args.display_freq == 0:
            progress.display(idx)
            


def val(args: edict, model: nn.Module, data_loader: DataLoader, criterion:SRNLoss, alphabet, epoch):
    '''验证模型'''
    batch_time = AverageMeter('BatchTime', ':.3f')
    loss = AverageMeter('Loss', ':.3e')
    img_loss = AverageMeter('ImageLoss', ':.3e')
    word_loss = AverageMeter('WordLoss', ':.3e')
    acc = AverageMeter('Acc', ':.3f')
    norm_edit_dist = AverageMeter('NormEditDist', ':.3f')
    progress = ProgressMeter(len(data_loader),
                             [batch_time, loss, img_loss, word_loss, acc, norm_edit_dist],
                             prefix='Test: ')
    print('validate model...')
    model.eval() # convert to evaluate mode.
    metric = RecMetric()
    
    with torch.no_grad():
        st = time.time()
        for idx, data in enumerate(data_loader):
            if args.gpu:
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.cuda()
            preds = model(data['image'], data, data['label'])
            losses = criterion(preds, losses['label'])
            # should be decode first.
            
            batch_indexs = torch.argmax(preds['predict'], dim=-1).reshape(-1, alphabet.max_len)
            
            m = metric(alphabet.decode_batch(batch_indexs), 
                    alphabet.decode_batch(data['label']))
            
            loss.update(losses['loss'].item())
            img_loss.update(losses['img_loss'].item())
            word_loss.update(losses['word_loss'].item())
            norm_edit_dist.update(m['norm_edit_dist'])
            acc.update(m['acc'])
            
            
            batch_time.update(time.time() - st)
            st = time.time()
            if idx % args.display_freq == 0:
                progress.display(idx)
                
        return metric.get_metric()
        
    


def save_checkpoint_srn(state, args, is_best=False, name='ckpt.pth.tar', best_name='best.pth.tar'):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('Create folder: {}'.format(args.save_dir))
    ckpt_path = os.path.join(args.save_dir, '{}_{}'.format(args.name, name))
    best_path = os.path.join(args.save_dir, '{}_{}'.format(args.name, best_name))
    save_checkpoint(state, ckpt_path, is_best, best_path)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', default='configs/demo.yaml',
                           help='the path of specified config.')
    args = argparser.parse_args()
    print(args)
    args = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    print(json.dumps(args, indent=2))
    main(args)
    
    
#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

# builtins
import os, argparse, time
from pathlib import Path
# project dependencies
from params import par
from model import DeepVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info
# external dependencies
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

## Parse passed Arguments
argparser = argparse.ArgumentParser(description="DeepVO Training")
argparser.add_argument('model_out', type=str, help="path where trained model will be saved (it will be named \'[--run_name]{_train,_valid}.model\')")
argparser.add_argument('dataset', type=str, help="dataset base directory")
argparser.add_argument('epochs', type=int, help="number epochs to train")
argparser.add_argument('batch_size', type=int, help="batch size for training (good results for 8 so far)")
argparser.add_argument('train_sequences', type=str, nargs='+', help="video indices from [dataset] used for training")
argparser.add_argument('--valid_sequences', '-vseq', type=str, default=None, nargs='+', help="video indices from [dataset] used for validation, if not set validation data will be created from [train_sequences] by partitioning using the value of [--partition]")
argparser.add_argument('--valid_sequences2', '-vseq2', type=str, default=None, nargs='+', help="video indices from [--dataset2] used for second validation, intended to be used if training should additionally be validated on alternative dataset")
argparser.add_argument('--dataset2', '-ds2', type=str, default=None, help="base directory of dataset used for second validation, intended to be used if training should additionally be validated on alternative dataset")
argparser.add_argument('--run_name', '-run', type=str, default='test_run', help="will be used for logging and naming of saved data (default: test_run)")
argparser.add_argument('--pretrained_flownet', '-flownet', type=str, default=None, help="pretrained flownet weights, if not set weights will be initialized randomly, will be ignored if [--resume] is set")
argparser.add_argument('--log_dir', '-log', type=str, default='logs', help="directory where log data will be saved")
argparser.add_argument('--model_load_path', '-load_model', type=str, default=None, help="model to be loaded if training is resumed")
argparser.add_argument('--optimizer_load_path', '-load_optim', type=str, default=None, help="optimizer to be loaded if training is resumed")
argparser.add_argument('--optimizer_save_path', '-save_optim', type=str, default=None, help="path where optimizer will be saved (it will be named \'[--run_name]{_train,_valid}.optimizer\')")
argparser.add_argument('--resume', '-resume', action='store_true', help="If set training will resume from model given by \'--model_load_path\' and \'--optimizer_load_path\'.")
argparser.add_argument('--start_epoch', '-ep', type=int, default=0, help="specify where to start counting the epochs, only used when \'--resume\' is set (default: 0)")
argparser.add_argument('--partition', '-p', type=float, default=0.8, help="set to number in range [0,1] to split train sequences into [-p]%% sequences for training and (1-[-p])%% for validation, will be ignored if [--valid_sequences] is set (default: 0.8)")
argparser.add_argument('--n_processors', '-np', type=int, default=4, help="number of processes to be invoked for dataset loading during training (default: 4)")
args = argparser.parse_args()

## Handle Arguments
# if training is resumed model_load_path and optimizer_load_path must be set too
if args.resume and not (args.model_load_path and args.optimizer_load_path):
    print('[ERROR] if [--resume] flag is set both, [--model_load_path] and [--optimizer_load_path] must be set too')
    exit()
# create required directory structure for model_save_path
model_file = args.run_name + '.model'
model_base = args.model_out
Path(model_base).mkdir(parents=True, exist_ok=True)
# create required directory structure for optimizer_save_path
optimizer_file = args.run_name + '.optimizer'
if args.optimizer_save_path: # case: optimizer_save_path is set -> use it
    optimizer_base = args.optimizer_save_path
    Path(optimizer_base).mkdir(parents=True, exist_ok=True)
else: # case: create optimizer_save_path not set -> use same dir same dir same dir as for model
    optimizer_base = model_base
# make log dir if not exist
Path(args.log_dir).mkdir(parents=True, exist_ok=True)
# set dataset dirs
image_dir = os.path.join(args.dataset, 'images')
pose_dir = os.path.join(args.dataset, 'poses_gt')
# if valid_sequences are passed set partition to None so that datasets will be created from train and valid sequences
args.partition = 0 if args.valid_sequences else args.partition
# if second validation is requested both args, -ds2 and -vseq2 must be set
if bool(args.dataset2) ^ bool(args.valid_sequences2):
    print('[ERROR] [--dataset2] and [--valid_sequences2] must both be set if you want to use second validation, else set none of them')
    exit()
use_second_validation = bool(args.dataset2) and bool(args.valid_sequences2)


## Prepare Data
print('subdivide trajectories into subsequences of random lengths between {} and {}'.format(par.seq_len[0], par.seq_len[1]))
if args.partition > 0: # case: create validation dataset by partitioning the train dataset
    print('make train data from sequences: {} (dataset: {})'.format(args.train_sequences, args.dataset))
    print('make validation data from training sequences by partitioning (p={})'.format(args.partition))
    train_df, valid_df = get_partition_data_info(image_dir, pose_dir, args.partition, args.train_sequences, par.seq_len, overlap=1, sample_times=par.sample_times, shuffle=True, sort=True)
else: # case: create training and validtion dataset from given list of sequences
    print('make train data from sequences: {} (dataset: {})'.format(args.train_sequences, args.dataset))
    train_df = get_data_info(image_dir, pose_dir, folder_list=args.train_sequences, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
    print('make validation data from sequences: {} (dataset: {})'.format(args.valid_sequences, args.dataset))
    valid_df = get_data_info(image_dir, pose_dir, folder_list=args.valid_sequences, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)

print('Create Dataset Loaders')
train_sampler = SortedRandomBatchSampler(train_df, args.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5) # NOTE why swap h and w?
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.n_processors, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, args.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5) # NOTE why swap h and w?
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=args.n_processors, pin_memory=par.pin_mem)

# make second validtion dataset
if use_second_validation:
    print('make second validation data from sequences: {} (dataset: {})'.format(args.valid_sequences2, args.dataset2))
    valid_df2 = get_data_info(os.path.join(args.dataset2, 'images'), os.path.join(args.dataset2, 'poses_gt'), folder_list=args.valid_sequences2, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
    valid_sampler2 = SortedRandomBatchSampler(valid_df2, args.batch_size, drop_last=True)
    valid_dataset2 = ImageSequenceDataset(valid_df2, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5) # NOTE why swap h and w?
    valid_dl2 = DataLoader(valid_dataset2, batch_sampler=valid_sampler2, num_workers=args.n_processors, pin_memory=par.pin_mem)

print('Number of samples in training dataset: ', len(train_df.index))
print('Number of samples in validation dataset: ', len(valid_df.index))
print('Number of samples in second validation dataset: ', len(valid_df2.index)) if use_second_validation else None
print('='*50)


## Prepare Model
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA enabled')
    M_deepvo = M_deepvo.cuda()
else:
    print('CUDA disabled')

## Load FlowNet weights
# NOTE the pretrained model assumes image rgb values in range [-0.5, 0.5]
if args.pretrained_flownet and not args.resume:
    print('load conv weights from pretrained FlowNet model ({})'.format(args.pretrained_flownet))
    if use_cuda:
        pretrained_w = torch.load(args.pretrained_flownet)
    else:
        pretrained_w = torch.load(args.pretrained_flownet, map_location='cpu')
    # Use conv-layer-weights of FlowNet for DeepVO conv-layers
    model_dict = M_deepvo.state_dict()
    update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    M_deepvo.load_state_dict(model_dict)
elif not args.pretrained_flownet and not args.resume:
    print('conv weights will be initialized randomly')
else:
    print('resume training, weights will be initialized from {}'.format(args.model_load_path))

## Create Optimizer
if par.optim['opt'] == 'Adam':
    optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=0.001, betas=(0.9, 0.999))
elif par.optim['opt'] == 'Adagrad':
    optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'])
elif par.optim['opt'] == 'Cosine':
    optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=par.optim['lr'])
    T_iter = par.optim['T']*len(train_dl)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

## Load trained DeepVO Model and Optimizer when resume Training
if args.resume:
    print('Load Model from: ', args.model_load_path)
    M_deepvo.load_state_dict(torch.load(args.model_load_path))
    print('Load Optimizer from: ', args.optimizer_load_path)
    optimizer.load_state_dict(torch.load(args.optimizer_load_path))

## Setup Logging
tb_dir = os.path.join(args.log_dir, 'tensorboard', args.run_name)
tb = SummaryWriter(log_dir=tb_dir)
# _ttype = torch.cuda.FloatTensor if use_cuda else torch.float32
# tb.add_graph(M_deepvo, torch.zeros((args.batch_size, int(sum(par.seq_len)/2), 3, par.img_w, par.img_h), dtype=_ttype))
print('TensorBoard will log to: {}'.format(tb_dir))

## Train Loop
min_loss_t  = 1e10
min_loss_v  = 1e10
min_loss_v2 = 1e10
M_deepvo.train()
epochs = range(args.epochs) if not args.resume else range(args.start_epoch, args.start_epoch + args.epochs)
for ep in epochs:
    st_t = time.time()
    print('='*50)
    print('epoch {}/{}'.format(ep+1, args.epochs))

    # train model
    M_deepvo.train()
    loss_mean = 0
    for it, (_, t_x, t_y) in enumerate(train_dl):
        print('train batch {}/{}, '.format(it+1, len(train_dl)), end='', flush=True)
        if use_cuda:
            t_x = t_x.cuda(non_blocking=par.pin_mem)
            t_y = t_y.cuda(non_blocking=par.pin_mem)
        ls = M_deepvo.step(t_x, t_y, optimizer).data.cpu().numpy()
        print('loss:', float(ls))
        loss_mean += float(ls)
        if par.optim == 'Cosine':
            lr_scheduler.step()
    loss_mean /= len(train_dl)
    tb.add_scalar('Loss/train', loss_mean, ep) # log ep train loss with tensorboard
    print('training took {:.1f} sec, mean loss: {}'.format(time.time()-st_t, loss_mean))

    # validate model
    st_t = time.time()
    M_deepvo.eval()
    loss_mean_valid = 0
    for it, (_, v_x, v_y) in enumerate(valid_dl):
        print('valid batch {}/{}, '.format(it+1, len(valid_dl)), end='', flush=True)
        if use_cuda:
            v_x = v_x.cuda(non_blocking=par.pin_mem)
            v_y = v_y.cuda(non_blocking=par.pin_mem)
        v_ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
        print('loss:', float(v_ls))
        loss_mean_valid += float(v_ls)
    loss_mean_valid /= len(valid_dl)
    tb.add_scalar('Loss/valid', loss_mean_valid, ep) # log ep valid loss with tensorboard
    print('validation took {:.1f} sec, mean loss: {}'.format(time.time()-st_t, loss_mean_valid))

    # validate model second time if requested
    if use_second_validation:
        st_t = time.time()
        M_deepvo.eval()
        loss_mean_valid2 = 0
        for it, (_, v_x, v_y) in enumerate(valid_dl2):
            print('2nd valid batch {}/{}, '.format(it+1, len(valid_dl2)), end='', flush=True)
            if use_cuda:
                v_x = v_x.cuda(non_blocking=par.pin_mem)
                v_y = v_y.cuda(non_blocking=par.pin_mem)
            v_ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
            print('loss:', float(v_ls))
            loss_mean_valid2 += float(v_ls)
        loss_mean_valid2 /= len(valid_dl2)
        tb.add_scalar('Loss/2nd valid', loss_mean_valid2, ep) # log ep valid loss with tensorboard
        print('2nd validation took {:.1f} sec, mean loss: {}'.format(time.time()-st_t, loss_mean_valid2))

    ## save models

    # save model if valid loss decreases
    check_interval = 1
    if loss_mean_valid < min_loss_v and ep % check_interval == 0:
        min_loss_v = loss_mean_valid
        print('Save model at ep {}, mean of valid loss: {}'.format(ep+1, loss_mean_valid))
        _model_save = os.path.join(model_base, os.path.splitext(model_file)[0] + '_valid' + os.path.splitext(model_file)[1])
        _optim_save = os.path.join(optimizer_base, os.path.splitext(optimizer_file)[0] + '_valid' + os.path.splitext(optimizer_file)[1])
        torch.save(M_deepvo.state_dict(), _model_save)
        torch.save(optimizer.state_dict(), _optim_save)
        tb.add_scalar('Checkpoints/valid', loss_mean_valid, ep)
    else:
        tb.add_scalar('Checkpoints/valid', 0.0, ep)

    if use_second_validation:
        # save model if 2nd valid loss decreases
        check_interval = 1
        if loss_mean_valid2 < min_loss_v2 and ep % check_interval == 0:
            min_loss_v2 = loss_mean_valid2
            print('Save model at ep {}, mean of 2nd valid loss: {}'.format(ep+1, loss_mean_valid2))
            _model_save = os.path.join(model_base, os.path.splitext(model_file)[0] + '_valid2' + os.path.splitext(model_file)[1])
            _optim_save = os.path.join(optimizer_base, os.path.splitext(optimizer_file)[0] + '_valid2' + os.path.splitext(optimizer_file)[1])
            torch.save(M_deepvo.state_dict(), _model_save)
            torch.save(optimizer.state_dict(), _optim_save)
            tb.add_scalar('Checkpoints/2nd valid', loss_mean_valid, ep)
        else:
            tb.add_scalar('Checkpoints/2nd valid', 0.0, ep)

    # save model if training loss decreases
    check_interval = 1
    if loss_mean < min_loss_t and ep % check_interval == 0:
        min_loss_t = loss_mean
        print('Save model at ep {}, mean of train loss: {}'.format(ep+1, loss_mean))
        _model_save = os.path.join(model_base, os.path.splitext(model_file)[0] + '_train' + os.path.splitext(model_file)[1])
        _optim_save = os.path.join(optimizer_base, os.path.splitext(optimizer_file)[0] + '_train' + os.path.splitext(optimizer_file)[1])
        torch.save(M_deepvo.state_dict(), _model_save)
        torch.save(optimizer.state_dict(), _optim_save)
        tb.add_scalar('Checkpoints/train', loss_mean, ep)
    else:
        tb.add_scalar('Checkpoints/train', 0.0, ep)

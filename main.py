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
argparser.add_argument('model_out', type=str, help="path where trained model will be saved")
argparser.add_argument('--run_name', '-run', type=str, default='test_run', help="name of experiment used for logging and naming saved data")
argparser.add_argument('--model_load_path', '-load_model', type=str, default=None, help="path from where model will be loaded if training is resumed")
argparser.add_argument('--optimizer_load_path', '-load_optim', type=str, default=None, help="path from where optimizer will be loaded if training is resumed")
argparser.add_argument('--optimizer_save_path', '-save_optim', type=str, default=None, help="path where optimizer will be saved")
argparser.add_argument('--resume', '-resume', action='store_true', help="If set training will resume from model given by \'--model_load_path\' and \'--optimizer_load_path\'.")
argparser.add_argument('--log_dir', '-log', type=str, default='logs', help="directory where log data should be saved")
args = argparser.parse_args()

## Check Arguments
# check if resume flag is set consistently
if args.resume and not (args.model_load_path and args.optimizer_load_path):
    print('[ERROR] if \'--resume\' flag is set both \'--model_load_path\' and \'--optimizer_load_path\' must be set')
    exit()
# create required directory structure for model_save_path
model_file = args.run_name + '.model'
model_base = args.model_out
Path(model_base).mkdir(parents=True, exist_ok=True)
# create required directory structure for optimizer_save_path
optimizer_file = args.run_name + '.optimizer'
if args.optimizer_save_path: # case: optimizer_save_path is set
    optimizer_base = args.optimizer_save_path
    Path(optimizer_base).mkdir(parents=True, exist_ok=True)
else: # case: create optimizer_save_path from model_save_path: Path/To/model.file -> Path/To/model_optimizer.file
    optimizer_base = model_base
# create required directory structure for logging
Path(args.log_dir).mkdir(parents=True, exist_ok=True)

## Prepare Data
print('Subdivide Trajectories into Subsequences of Lengths between {} and {}'.format(par.seq_len[0], par.seq_len[1]))
if par.partition != None:
    train_df, valid_df = get_partition_data_info(par.partition, par.train_seq, par.seq_len, overlap=1, sample_times=par.sample_times, shuffle=True, sort=True)
else:
    train_df = get_data_info(folder_list=par.train_seq, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
    valid_df = get_data_info(folder_list=par.valid_seq, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)

print('Create Dataset Loaders')
train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5) # NOTE why swap h and w?
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5) # NOTE why swap h and w?
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

print('Number of samples in training dataset: ', len(train_df.index))
print('Number of samples in validation dataset: ', len(valid_df.index))
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
if par.pretrained_flownet and not args.resume:
    print('Load pretrained FlowNet weights')
    if use_cuda:
        pretrained_w = torch.load(par.pretrained_flownet)
    else:
        pretrained_w = torch.load(par.pretrained_flownet, map_location='cpu')
    # Use conv-layer-weights of FlowNet for DeepVO conv-layers
    model_dict = M_deepvo.state_dict()
    update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    M_deepvo.load_state_dict(model_dict)

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
tb_dir = os.path.join(os.path.join(args.log_dir, 'tensorboard'), args.run_name)
tb = SummaryWriter(log_dir=tb_dir)
if use_cuda:
    tb.add_graph(M_deepvo, torch.zeros(par.batch_size, int(sum(par.seq_len)/2), 3, par.img_w, par.img_h, dtype=torch.cuda.FloatTensor))
else:
    tb.add_graph(M_deepvo, torch.zeros(par.batch_size, int(sum(par.seq_len)/2), 3, par.img_w, par.img_h, dtype=torch.float32))
print('TensorBoard will log to: {}'.format(tb_dir))

## Train Loop
min_loss_t = 1e10
min_loss_v = 1e10
M_deepvo.train()
for ep in range(par.epochs):
    st_t = time.time()
    print('='*50)
    print('epoch {}/{}'.format(ep+1, par.epochs))

    # train model
    M_deepvo.train()
    loss_mean = 0
    t_loss_list = []
    for it, (_, t_x, t_y) in enumerate(train_dl):
        print('train batch {}/{}, '.format(it+1, len(train_dl)), end='', flush=True)
        if use_cuda:
            t_x = t_x.cuda(non_blocking=par.pin_mem)
            t_y = t_y.cuda(non_blocking=par.pin_mem)
        ls = M_deepvo.step(t_x, t_y, optimizer).data.cpu().numpy()
        print('loss:', float(ls))
        t_loss_list.append(float(ls))
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
    v_loss_list = []
    for it, (_, v_x, v_y) in enumerate(valid_dl):
        print('valid batch {}/{}, '.format(it+1, len(valid_dl)), end='', flush=True)
        if use_cuda:
            v_x = v_x.cuda(non_blocking=par.pin_mem)
            v_y = v_y.cuda(non_blocking=par.pin_mem)
        v_ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
        print('loss:', float(v_ls))
        v_loss_list.append(float(v_ls))
        loss_mean_valid += float(v_ls)
    loss_mean_valid /= len(valid_dl)
    tb.add_scalar('Loss/valid', loss_mean_valid, ep) # log ep valid loss with tensorboard
    print('validation took {:.1f} sec, mean loss: {}'.format(time.time()-st_t, loss_mean_valid))

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

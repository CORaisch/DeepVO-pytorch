import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, argparse
import time
import pandas as pd
from params import par
from model import DeepVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info

# parse passed arguments
argparser = argparse.ArgumentParser(description="DeepVO Training")
argparser.add_argument('--remote_dir', '-remote', type=str, default=None, help="If train on cluster set this to the remote directory like \'/scratch/X\'. All datasets used for training will be copied to this directory.")
argparser.add_argument('--home_dir', '-home', type=str, default=None, help="If train on cluster set this to the home directory. Data like models and weights will be read from there and checkpoints will be written to there too.")
argparser.add_argument('--resume', '-resume', action='store_true', help="If set training will resume from given model.")
args = argparser.parse_args()
# update directories when executing on cluster
par.set_remote_dir(args.remote_dir)
par.set_home_dir(args.home_dir)
par.set_resume(args.resume)

# Write all hyperparameters to record_path
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
    f.write('\n'+'='*50 + '\n')
    f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
    f.write('\n'+'='*50 + '\n')

# Prepare Data
print('Subsequence Trajectories')
if par.partition != None:
    partition = par.partition
    train_df, valid_df = get_partition_data_info(partition, par.train_video, par.seq_len, overlap=1, sample_times=par.sample_times, shuffle=True, sort=True)
else:
    train_df = get_data_info(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
    valid_df = get_data_info(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)

print('Create Dataset')
train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
# train_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_h, par.img_w), par.img_means, par.img_stds, par.minus_point_5) # NOTE why swap h and w ?
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
# valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_h, par.img_w), par.img_means, par.img_stds, par.minus_point_5) # NOTE why swap h and w ?
valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

print('Number of samples in training dataset: ', len(train_df.index))
print('Number of samples in validation dataset: ', len(valid_df.index))
print('='*50)


# Model
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA enabled')
    M_deepvo = M_deepvo.cuda()


# Load FlowNet weights pretrained with FlyingChairs
# NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
if par.pretrained_flownet and not par.resume:
    if use_cuda:
        pretrained_w = torch.load(par.pretrained_flownet)
    else:
        pretrained_w = torch.load(par.pretrained_flownet, map_location='cpu')
    print('Load FlowNet pretrained model')
    # Use only conv-layer-part of FlowNet as CNN for DeepVO
    model_dict = M_deepvo.state_dict()
    update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    M_deepvo.load_state_dict(model_dict)


# Create optimizer
if par.optim['opt'] == 'Adam':
    optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=0.001, betas=(0.9, 0.999))
elif par.optim['opt'] == 'Adagrad':
    optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'])
elif par.optim['opt'] == 'Cosine':
    optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=par.optim['lr'])
    T_iter = par.optim['T']*len(train_dl)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

# Load trained DeepVO model and optimizer
if par.resume:
    M_deepvo.load_state_dict(torch.load(par.load_model_path))
    optimizer.load_state_dict(torch.load(par.load_optimizer_path))
    print('Load model from: ', par.load_model_path)
    print('Load optimizer from: ', par.load_optimizer_path)


# setup logging
if args.home_dir:
    tb_dir = os.path.join(args.home_dir, os.path.join('records/tensorboard', par.experiment_name.split('/')[1]))
else:
    tb_dir = os.path.join('records/tensorboard', par.experiment_name.split('/')[1])
tb = SummaryWriter(log_dir=tb_dir)
tb.add_graph(M_deepvo, torch.zeros(par.batch_size, int(sum(par.seq_len)/2), 3, par.img_w, par.img_h, dtype=torch.float32))
print('tensorboard log dir:', tb_dir)
print('Record loss in: ', par.record_path)

# Train
min_loss_t = 1e10
min_loss_v = 1e10
M_deepvo.train()
for ep in range(par.epochs):
    st_t = time.time()
    print('='*50)
    print('epoch {}/{}'.format(ep+1, par.epochs))
    # Train
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

    # Validation
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


    with open(par.record_path, 'a') as f:
        f.write('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))
    print('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))

    # Save model

    # save if the valid loss decrease
    check_interval = 1
    if loss_mean_valid < min_loss_v and ep % check_interval == 0:
        min_loss_v = loss_mean_valid
        print('Save model at ep {}, mean of valid loss: {}'.format(ep+1, loss_mean_valid))  # use 4.6 sec
        torch.save(M_deepvo.state_dict(), par.save_model_path+'.valid')
        torch.save(optimizer.state_dict(), par.save_optimzer_path+'.valid')
        tb.add_scalar('Checkpoints/valid', loss_mean_valid, ep)
    else:
        tb.add_scalar('Checkpoints/valid', 0.0, ep)

    # save if the training loss decrease
    check_interval = 1
    if loss_mean < min_loss_t and ep % check_interval == 0:
        min_loss_t = loss_mean
        print('Save model at ep {}, mean of train loss: {}'.format(ep+1, loss_mean))
        torch.save(M_deepvo.state_dict(), par.save_model_path+'.train')
        torch.save(optimizer.state_dict(), par.save_optimzer_path+'.train')
        tb.add_scalar('Checkpoints/train', loss_mean, ep)
    else:
        tb.add_scalar('Checkpoints/train', 0.0, ep)


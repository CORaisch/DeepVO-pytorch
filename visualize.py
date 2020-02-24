#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

# builtins
import os, time, argparse
from pathlib import Path
# project dependencies
from params import par
# external dependencies
import numpy as np
import matplotlib.pyplot as plt

def plot_route(gt, out, c_gt='g', c_out='r'):
    x_idx = 3
    y_idx = 5
    x = [v for v in gt[:, x_idx]]
    y = [v for v in gt[:, y_idx]]
    plt.plot(x, y, color=c_gt, label='Ground Truth')
    #plt.scatter(x, y, color='b')

    x = [v for v in out[:, x_idx]]
    y = [v for v in out[:, y_idx]]
    plt.plot(x, y, color=c_out, label='DeepVO')
    #plt.scatter(x, y, color='b')
    plt.gca().set_aspect('equal', adjustable='datalim')

def route_to_png_colorized(gt, out, path, step=200):
    '''plot gradient color'''
    plt.clf()
    plt.scatter([gt[0][3]], [gt[0][5]], label='sequence start', marker='s', color='k')
    for st in range(0, len(out), step):
        end = st + step
        g = max(0.2, st/len(out))
        c_gt = (0, g, 0)
        c_out = (1, g, 0)
        plot_route(gt[st:end], out[st:end], c_gt, c_out)
        if st == 0:
            plt.legend()
        plt.title('Video {}'.format(seq))
        save_name = os.path.join(os.path.join(path, 'color'), 'route_{}.png'.format(seq))
    plt.savefig(save_name)

def route_to_png_plain(gt, out, path, step=200):
    '''plot one color'''
    plt.clf()
    plt.scatter([gt[0][3]], [gt[0][5]], label='sequence start', marker='s', color='k')
    plot_route(gt, out, 'r', 'b')
    plt.legend()
    plt.title('Video {}'.format(seq))
    save_name = os.path.join(os.path.join(path, 'plain'), 'route_{}.png'.format(seq))
    plt.savefig(save_name)

# parse passed arguments
argparser = argparse.ArgumentParser(description="DeepVO Visualization")
argparser.add_argument('src', type=str, help="path from where DeepVO trajectories will be read")
argparser.add_argument('dst', type=str, help="path where images will be saved")
argparser.add_argument('--gt_poses', '-gt', type=str, default=None, help="path to GT poses files, if not set it will be read from params")
argparser.add_argument('--sequences', '-seq', type=str, default=None, nargs='+', help="list of videos (indices) used for preprocessing, if not set it will be read from params")
argparser.add_argument('--gradient_color', '-gradient', action='store_true', help="when set trajectory will be colorized depending on local error.")
argparser.add_argument('--both', '-both', action='store_true', help="when set trajectory will be saved in colorized and plain version.")
args = argparser.parse_args()

# setup directory structure
if args.both or not args.gradient_color:
    Path(os.path.join(args.dst, 'plain')).mkdir(parents=True, exist_ok=True)
if args.both or args.gradient_color:
    Path(os.path.join(args.dst, 'color')).mkdir(parents=True, exist_ok=True)

# Load GT and predicted poses
if args.gt_poses and args.sequences: # case: take paths from args
    print('[INFO] GT poses and video sequences to visualize will be read from arguments')
    gt_poses = args.gt_poses
    sequences = args.sequences
    pass
elif args.gt_poses ^ args.sequences: # case: both args must be set to read from args
    print('[ERROR] \'--gt_poses\' and \'--sequences\' must be both set or none of both is set')
    exit()
else:
    print('[INFO] GT poses and video sequences to visualize will be read from params')
    gt_poses = par.pose_dir
    sequences = par.train_seq + list(set(par.valid_seq) - set(par.train_seq)) # NOTE train_video ∪ valid_video, i.e. removing duplicates if exist

for seq in sequences:
    print('='*50)
    print('Sequence {}'.format(seq))

    gt_pose_path = os.path.join(gt_poses, seq + '.npy')
    gt = np.load(gt_pose_path)
    est_path = os.path.join(args.src, 'out_{}.txt'.format(seq))
    try:
        with open(est_path, 'r') as f:
            # read in estimated trajectory
            est = [l.split('\n')[0] for l in f.readlines()]
            for i, line in enumerate(est):
                est[i] = [float(v) for v in line.split(',')]
            est = np.array(est)

            # compute errors
            mse_rotate = np.mean((est[:, :3] - gt[:, :3])**2)
            mse_translate = np.mean((est[:, 3:] - gt[:, 3:6])**2)
            print('mse_rotate: ', mse_rotate)
            print('mse_translate: ', mse_translate)

            # draw images
            if args.both or not args.gradient_color:
                route_to_png_plain(gt, est, args.dst)
            if args.both or args.gradient_color:
                route_to_png_colorized(gt, est, args.dst)
    except FileNotFoundError:
        print('[WARNING] file \'{}\' does not exist, it will be skipped'.format(est_path))



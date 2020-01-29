import matplotlib.pyplot as plt
import numpy as np
import os, time, argparse
from params import par

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

def route_to_png_colorized(gt, out, step=200):
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
        plt.title('Video {}'.format(video))
        save_name = os.path.join(os.path.join(predicted_result_dir, 'colorized'), 'route_{}.png'.format(video))
    plt.savefig(save_name)

def route_to_png_plain(gt, out, step=200):
    '''plot one color'''
    plt.clf()
    plt.scatter([gt[0][3]], [gt[0][5]], label='sequence start', marker='s', color='k')
    plot_route(gt, out, 'r', 'b')
    plt.legend()
    plt.title('Video {}'.format(video))
    save_name = os.path.join(os.path.join(predicted_result_dir, 'plain'), 'route_{}.png'.format(video))
    plt.savefig(save_name)

# parse passed arguments
argparser = argparse.ArgumentParser(description="DeepVO Visualization")
argparser.add_argument('--results_dir', '-results', type=str, default=None, help="directory where the results are stored.")
argparser.add_argument('--gradient_color', '-gradient', action='store_true', help="when set trajectory will be colorized depending on local error.")
argparser.add_argument('--both', '-both', action='store_true', help="when set trajectory will be saved in colorized and plain version.")
args = argparser.parse_args()

# setup directories and config
pose_GT_dir = par.pose_dir
predicted_result_dir = args.results_dir
if args.both or not args.gradient_color:
    if not os.path.isdir(os.path.join(predicted_result_dir, 'plain')):
        os.makedirs(os.path.join(predicted_result_dir, 'plain'))
if args.both or args.gradient_color:
    if not os.path.isdir(os.path.join(predicted_result_dir, 'colorized')):
        os.makedirs(os.path.join(predicted_result_dir, 'colorized'))


# Load GT and predicted poses
video_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

for video in video_list:
    print('='*50)
    print('Video {}'.format(video))

    GT_pose_path = '{}{}.npy'.format(pose_GT_dir, video)
    gt = np.load(GT_pose_path)
    pose_result_path = os.path.join(predicted_result_dir, 'out_{}.txt'.format(video))
    with open(pose_result_path) as f_out:
        out = [l.split('\n')[0] for l in f_out.readlines()]
        for i, line in enumerate(out):
            out[i] = [float(v) for v in line.split(',')]
        out = np.array(out)
        mse_rotate = 100 * np.mean((out[:, :3] - gt[:, :3])**2)
        mse_translate = np.mean((out[:, 3:] - gt[:, 3:6])**2)
        print('mse_rotate: ', mse_rotate)
        print('mse_translate: ', mse_translate)


    # draw images
    if args.both or not args.gradient_color:
        route_to_png_plain(gt, out)
    if args.both or args.gradient_color:
        route_to_png_colorized(gt, out)

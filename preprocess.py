#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

# builtins
import os, argparse, glob, time, math
from PIL import Image
# project dependencies
from params import par
from helper import R_to_angle
# external dependencies
import numpy as np
import torch
from torchvision import transforms

def clean_unused_kitti_images():
    seq_frame = {'00': ['000', '004540'],
                '01': ['000', '001100'],
                '02': ['000', '004660'],
                '03': ['000', '000800'],
                '04': ['000', '000270'],
                '05': ['000', '002760'],
                '06': ['000', '001100'],
                '07': ['000', '001100'],
                '08': ['001100', '005170'],
                '09': ['000', '001590'],
                '10': ['000', '001200']
                }
    for dir_id, img_ids in seq_frame.items():
        dir_path = '{}{}/'.format(par.image_dir, dir_id)
        if not os.path.exists(dir_path):
            continue

        print('Cleaning {} directory'.format(dir_id))
        start, end = img_ids
        start, end = int(start), int(end)
        for idx in range(0, start):
            img_name = '{:010d}.png'.format(idx)
            img_path = '{}{}/{}'.format(par.image_dir, dir_id, img_name)
            if os.path.isfile(img_path):
                os.remove(img_path)
        for idx in range(end+1, 10000):
            img_name = '{:010d}.png'.format(idx)
            img_path = '{}{}/{}'.format(par.image_dir, dir_id, img_name)
            if os.path.isfile(img_path):
                os.remove(img_path)


# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data(sequences):
    start_t = time.time()
    for seq in sequences:
        fn = '{}.txt'.format(os.path.join(par.pose_dir, seq))
        if not os.path.exists(fn):
            continue
        print('Transforming {}...'.format(fn))
        with open(fn) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]
            poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines]    # list of pose (pose=list of 12 floats)
            poses = np.array(poses)
            base_fn = os.path.splitext(fn)[0]
            np.save(base_fn+'.npy', poses)
            print('Sequence {}: shape={}'.format(seq, poses.shape))
    print('elapsed time = {}'.format(time.time()-start_t))

def calculate_rgb_mean_std(image_path_list, minus_point_5=False):
    n_images = len(image_path_list)
    cnt_pixels = 0
    print('Numbers of frames in dataset: {}'.format(n_images))
    mean_np = [0, 0, 0]
    mean_tensor = [0, 0, 0]
    to_tensor = transforms.ToTensor()

    image_sequence = []
    for idx, img_path in enumerate(image_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        if par.grayscale:
            img_as_img = transforms.functional.to_grayscale(img_as_img, num_output_channels=3)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        cnt_pixels += img_as_np.shape[1]*img_as_np.shape[2]
        for c in range(3):
            mean_tensor[c] += float(torch.sum(img_as_tensor[c]))
            mean_np[c] += float(np.sum(img_as_np[c]))
    mean_tensor =  [v / cnt_pixels for v in mean_tensor]
    mean_np = [v / cnt_pixels for v in mean_np]
    print('mean_tensor = ', mean_tensor)
    print('mean_np = ', mean_np)

    std_tensor = [0, 0, 0]
    std_np = [0, 0, 0]
    for idx, img_path in enumerate(image_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        if par.grayscale:
            img_as_img = transforms.functional.to_grayscale(img_as_img, num_output_channels=3)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        for c in range(3):
            tmp = (img_as_tensor[c] - mean_tensor[c])**2
            std_tensor[c] += float(torch.sum(tmp))
            tmp = (img_as_np[c] - mean_np[c])**2
            std_np[c] += float(np.sum(tmp))
    std_tensor = [math.sqrt(v / cnt_pixels) for v in std_tensor]
    std_np = [math.sqrt(v / cnt_pixels) for v in std_np]
    print('std_tensor = ', std_tensor)
    print('std_np = ', std_np)


if __name__ == '__main__':
    # handle args
    argparser = argparse.ArgumentParser(description="DeepVO Preprocessing")
    argparser.add_argument('--kitti', '-kitti', action='store_true', help="set if preprocessing KITTI data, additional images will be removed for KITTI")
    argparser.add_argument('--dataset_dir', '-ds', type=str, default=None, help="directory of dataset, if not set it will be taken from params")
    argparser.add_argument('--sequences', '-seq', type=str, default=None, nargs='+', help="list of video sequences (indices) used for preprocessing, if not set it will be taken from params")
    args = argparser.parse_args()

    # set new dataset dir if requested
    if args.dataset_dir:
        par.data_dir = args.dataset_dir
        par.image_dir = os.path.join(par.data_dir, 'images/')
        par.pose_dir = os.path.join(par.data_dir, 'poses_gt/')

    # clean KITTI images as recommended by paper
    if args.kitti:
        clean_unused_kitti_images()

    # get sequences for preprocessing from params if not given as argument
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = par.train_seq + list(set(par.valid_seq) - set(par.train_seq)) # NOTE train_video ∪ valid_video, i.e. removing duplicates if exist

    # convert pose data
    create_pose_data(sequences)

    # if no laplace preprocessing, then preprocess by normalizing image inputs with mean and std (as done in DeepVO paper)
    if not par.laplace_preprocessing:
        # Calculate RGB means of images in training sequences
        image_path_list = []
        for folder in sequences:
            image_path_list += glob.glob('{}/*.png'.format(os.path.join(par.image_dir, folder)))
        calculate_rgb_mean_std(image_path_list, minus_point_5=par.minus_point_5)

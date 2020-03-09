#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

# builtins
import os, argparse, glob, time, math
from PIL import Image
# project dependencies
from helper import R_to_angle
# external dependencies
import numpy as np
import torch
from torchvision import transforms

def clean_unused_kitti_images(image_dir):
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
        dir_path = '{}{}/'.format(image_dir, dir_id)
        if not os.path.exists(dir_path):
            continue

        print('Cleaning {} directory'.format(dir_id))
        start, end = img_ids
        start, end = int(start), int(end)
        for idx in range(0, start):
            img_name = '{:010d}.png'.format(idx)
            img_path = '{}{}/{}'.format(image_dir, dir_id, img_name)
            if os.path.isfile(img_path):
                os.remove(img_path)
        for idx in range(end+1, 10000):
            img_name = '{:010d}.png'.format(idx)
            img_path = '{}{}/{}'.format(image_dir, dir_id, img_name)
            if os.path.isfile(img_path):
                os.remove(img_path)


# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data(sequences, pose_dir):
    start_t = time.time()
    for seq in sequences:
        fn = '{}.txt'.format(os.path.join(pose_dir, seq))
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

def calculate_rgb_mean_std(image_path_list, minus_point_5=False, grayscale=False):
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
        if grayscale:
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
        if grayscale:
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
    argparser = argparse.ArgumentParser(description="Dataset Preprocessing like proposed in DeepVO Paper, i.e. by computing Mean and STDEV Image)")
    argparser.add_argument('dataset', type=str, help="dataset base directory")
    argparser.add_argument('sequences', type=str, nargs='+', help="list of video sequences indices on which preprocessing should be computed")
    argparser.add_argument('--kitti', '-kitti', action='store_true', help="set if preprocessing KITTI data, additional images will be removed for KITTI")
    argparser.add_argument('--grayscale', '-gray', action='store_true', help="set if mean and std image should be computed on grayscale images")
    argparser.add_argument('--minus_point_5', '-mp5', action='store_true', help="set if pixel range should be shifted to [-0.5,0.5] before preprocessing")
    args = argparser.parse_args()

    # set dataset dir
    data_dir = args.dataset
    image_dir = os.path.join(data_dir, 'images/')
    pose_dir = os.path.join(data_dir, 'poses_gt/')

    # set sequences for preprocessing
    sequences = args.sequences

    # if preprocess KITTI, clean KITTI images as recommended by paper
    if args.kitti:
        clean_unused_kitti_images(image_dir)

    # convert pose data
    create_pose_data(sequences, pose_dir)

    # Calculate RGB means of images in training sequences
    image_path_list = []
    for folder in sequences:
        image_path_list += glob.glob(os.path.join(image_dir, folder, '*.png'))
    calculate_rgb_mean_std(image_path_list, minus_point_5=args.minus_point_5, grayscale=args.grayscale)

#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

# builtins
import glob, os, time, argparse
from math import ceil
from pathlib import Path
from PIL import Image
# project dependencies
from params import par
from model import DeepVO
from utils import eulerAnglesToRotationMatrix
from data_helper import get_data_info, ImageSequenceDataset
# external dependencies
import numpy as np
import torch
from torch.utils.data import DataLoader

def mat_to_string(mat):
    '''converts a pose matrix to string in KITTI format'''
    # NOTE expecting 3x4 or 4x4 pose matrix (SE(3))
    ret  = str(mat[0,0]) + " " + str(mat[0,1]) + " " + str(mat[0,2]) + " " + str(mat[0,3]) + " "
    ret += str(mat[1,0]) + " " + str(mat[1,1]) + " " + str(mat[1,2]) + " " + str(mat[1,3]) + " "
    ret += str(mat[2,0]) + " " + str(mat[2,1]) + " " + str(mat[2,2]) + " " + str(mat[2,3]) + "\n"
    return ret

def euler_to_mat(p):
    '''compose and return transformation matrix T from 6 params p'''
    t = np.matrix(p[3:]).reshape((3,1))
    R = np.matrix(eulerAnglesToRotationMatrix(p[:3]))
    T = np.matrix(np.eye(4, dtype=p.dtype)); T[:3,:3] = R; T[:3,3] = t;
    return T

# parse passed arguments
argparser = argparse.ArgumentParser(description="DeepVO Testing")
argparser.add_argument('model', type=str, help="path to trained model")
argparser.add_argument('out', type=str, help="path where estimates will be saved")
argparser.add_argument('dataset', type=str, help="dataset base directory")
argparser.add_argument('sequences', type=str, nargs='+', help="video indices to test on")
argparser.add_argument('--batch_size', '-bs', type=int, default=8, help="batch size for testing (default: 8)")
argparser.add_argument('--only_yaw', action='store_true', help="use only yaw prediction, rest is set to zero.")
args = argparser.parse_args()

if __name__ == '__main__':

    # set dataset to test on
    image_dir = os.path.join(args.dataset, 'images')
    pose_dir = os.path.join(args.dataset, 'poses_gt')

    # prepare directory structure
    load_model_path = args.model
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # load model
    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        M_deepvo = M_deepvo.cuda()
        M_deepvo.load_state_dict(torch.load(load_model_path))
    else:
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location={'cuda:0': 'cpu'}))
    print('load model from: ', load_model_path)

    # prepare dataset
    n_workers = 1
    seq_len = int((par.seq_len[0]+par.seq_len[1])/2)
    overlap = 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))

    # test loop
    for test_seq in args.sequences:
        # measure time
        st_t = time.time()

        # print expected output values
        n_poses = len(glob.glob(os.path.join(image_dir, test_seq, '*.png')))
        print('exp. #sub-sequences = {}, exp. #batches = {}'.format(n_poses-overlap, ceil((n_poses-overlap)/args.batch_size)))

        # make dataloader
        df = get_data_info(image_dir, pose_dir, folder_list=[test_seq], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        df = df.loc[df.seq_len == seq_len]  # drop last
        dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=n_workers)

        # loop over batched sub-sequences
        M_deepvo.eval()
        trajectory = [ np.matrix(np.eye(4, dtype=np.float)) ]
        n_batch = len(dataloader)

        for i, batch in enumerate(dataloader):
            print('{} / {}'.format(i, n_batch), end='\r', flush=True)

            # predict batched sequences of poses
            _, x, y = batch
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            pred_batch = M_deepvo.forward(x)
            # NOTE batch and pred_batch are tensors of rank Bx(S-1)x6

            # integrate poses
            pred_batch = pred_batch.data.cpu().numpy()
            for pred_seq in pred_batch:
                T0 = trajectory[-1]
                for pose in pred_seq:
                    # pose = pred_seq[-1]
                    # get relative pose
                    if args.only_yaw:
                        pose[0] = 0; pose[2] = 0;
                    T = euler_to_mat(pose)
                    # integrate abs pose
                    trajectory.append(T0*T)

        # print status
        delta_t = time.localtime(time.time() - st_t)
        print('len(trajectory):', len(trajectory))
        print('exp. len:', n_poses)
        print('delta t: {}:{} min'.format(delta_t.tm_min, delta_t.tm_sec))

        # save trajectory in KITTI format
        with open(os.path.join(args.out, 'out_{}.txt'.format(test_seq)), 'w') as f:
            # for pose in trajectory:
            for pose in trajectory:
                f.write(mat_to_string(pose))

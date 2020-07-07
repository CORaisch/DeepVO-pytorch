#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

# NOTE this file shows that the transformation from abs matrix to rel euler poses is not working properly and should be reimplemented in original code

# bultins
import sys, os, glob, argparse
sys.path.append('..')
from pathlib import Path
# project dependencies
from data_helper import get_data_info, ImageSequenceDataset
from utils import eulerAnglesToRotationMatrix, normalize_angle_delta, R_to_angle
# external dependencies
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PoseSequenceDataset(Dataset):
    '''loading gt poses same as with data_helper.ImageSequenceDataset but omitting the image loading'''

    def __init__(self, info_dataframe):
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def _to_euler(self, R):
        import math
        # NOTE code taken from: https://www.learnopencv.com/rotation-matrix-to-euler-angles
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    def _to_mat(self, arr, R):
        '''arr: [(r_0, r_1, r_2, t_0, t_1, t_2), R], |arr[0]| = 6, R: 3x3 matrix'''
        t = np.matrix(arr[3:]).reshape((3,1))
        T = np.matrix(np.eye(4, dtype=R.dtype))
        T[:3,:3] = R; T[:3,3] = t;
        return T

    def _inv(self, T):
        '''T: [R|t] as 4x4 matrix, inv: [R.T|-R.T*t] as 4x4 matrix'''
        Inv = np.matrix(np.eye(4, dtype=T.dtype))
        R = T[:3,:3]; t = T[:3,3]
        Inv[:3,:3] = R.T; Inv[:3,3] = -R.T*t;
        return Inv

    def __getitem__(self, index):
        seq_raw = np.hsplit(self.groundtruth_arr[index], np.array([6]))
        seq_len = seq_raw[0].shape[0]
        seq_abs = [ self._to_mat(seq_raw[0][i], seq_raw[1][i].reshape((3,3))) for i in range(seq_len) ]
        seq_rel = [ self._inv(seq_abs[i]) * seq_abs[i+1] for i in range(seq_len-1) ]
        seq_gt  = [ np.concatenate((self._to_euler(T[:3,:3]), np.asarray(T[:3,3])), axis=None) for T in seq_rel ]
        return torch.FloatTensor(seq_gt)

    def __len__(self):
        return len(self.data_info.index)

# parse arguments
argparser = argparse.ArgumentParser(description="transforms DVO poses as used for training back to 12 DoF absolute poses. Will be used to validate ds transformation.")
argparser.add_argument('out', type=str, help="path where results will be saved")
argparser.add_argument('dataset', type=str, help="dataset base directory")
argparser.add_argument('sequences', type=str, nargs='+', help="video indices to transform")
argparser.add_argument('--batch_size', '-bs', type=int, default=8, help="batch size for transforming (default: 8)")
args = argparser.parse_args()

def mat_to_string(mat):
    # NOTE expecting 3x4 or 4x4 pose matrix (SE(3))
    ret  = str(mat[0,0]) + " " + str(mat[0,1]) + " " + str(mat[0,2]) + " " + str(mat[0,3]) + " "
    ret += str(mat[1,0]) + " " + str(mat[1,1]) + " " + str(mat[1,2]) + " " + str(mat[1,3]) + " "
    ret += str(mat[2,0]) + " " + str(mat[2,1]) + " " + str(mat[2,2]) + " " + str(mat[2,3]) + "\n"
    return ret

if __name__ == '__main__':
    # set dataset to test on
    image_dir = os.path.join(args.dataset, 'images')
    pose_dir = os.path.join(args.dataset, 'poses_gt')

    # prepare directory structure
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # prepare dataset
    n_workers = 1
    seq_len = 2
    overlap = 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))

    # loop over sequences
    for seq in args.sequences:
        # create ds to iterate
        df = get_data_info(image_dir, pose_dir, folder_list=[seq], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        dataset = PoseSequenceDataset(df)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers)

        # loop over sequence
        trajectory = [ np.matrix(np.eye(4, dtype=np.float)) ]
        n_batch = len(dataloader)
        for i, batch in enumerate(dataloader):
            print('{} / {}'.format(i, n_batch), end='\r', flush=True)
            # NOTE batch: tensor of rank BSx5x6 (BS x SEQ_LEN-1 x 6 dof)
            for pred_seq in batch:
                for pred in pred_seq:
                    # compose T
                    p = pred.numpy()
                    t = np.matrix(p[3:]).reshape((3,1))
                    R = np.matrix(eulerAnglesToRotationMatrix(p[:3]))
                    T = np.matrix(np.eye(4, dtype=p.dtype)); T[:3,:3] = R; T[:3,3] = t;
                    # integrate abs pose
                    trajectory.append(trajectory[-1]*T)

        print('len(trajectory):', len(trajectory))
        print('expect len:', len(glob.glob(os.path.join(image_dir, seq, '*.png'))))

        # save trajectory in KITTI format
        with open(os.path.join(args.out, 'out_{}.txt'.format(seq)), 'w') as f:
            # for pose in trajectory:
            for pose in trajectory:
                f.write(mat_to_string(pose))

#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

import os, sys, argparse
from pathlib import Path
sys.path.append('..')
import numpy as np
from utils import eulerAnglesToRotationMatrix

def mat_to_string(mat):
    # NOTE expecting 3x4 or 4x4 np.matrix/np.ndarray
    ret  = str(mat[0,0]) + " " + str(mat[0,1]) + " " + str(mat[0,2]) + " " + str(mat[0,3]) + " "
    ret += str(mat[1,0]) + " " + str(mat[1,1]) + " " + str(mat[1,2]) + " " + str(mat[1,3]) + " "
    ret += str(mat[2,0]) + " " + str(mat[2,1]) + " " + str(mat[2,2]) + " " + str(mat[2,3]) + "\n"
    return ret

def to_mat(arr):
    '''arr: [r_0, r_1, r_2, t_0, t_1, t_2]'''
    assert len(arr)==6
    t = np.matrix(arr[3:]).reshape((3,1))
    R = eulerAnglesToRotationMatrix([arr[1], arr[0], arr[2]])
    T = np.matrix(np.eye(4, dtype=R.dtype))
    T[:3,:3] = R; T[:3,3] = t;
    return T

# parse passed arguments
argparser = argparse.ArgumentParser(description="DeepVO Visualization")
argparser.add_argument('base', type=str, help="path from where DeepVO trajectories will be read")
argparser.add_argument('out', type=str, help="path where converted trajectories will be saved")
argparser.add_argument('sequences', type=str, nargs='+', help="video indices to convert")
args = argparser.parse_args()

# plot estimates against gt
for seq in args.sequences:
    print('convert sequence {}...'.format(seq), flush=True, end='')

    Path(args.out).mkdir(parents=True, exist_ok=True)

    est_path = os.path.join(args.base, 'out_{}.txt'.format(seq))
    out_path = os.path.join(args.out, 'out_{}.txt'.format(seq))
    try:
        with open(est_path, 'r') as f:
            # read in estimated trajectory
            poses = [ l.split('\n')[0] for l in f.readlines() ]
            for i, line in enumerate(poses):
                poses[i] = [ float(v) for v in line.split(',') ]
            traj = [ to_mat(p) for p in poses ]
            traj_str = [ mat_to_string(p) for p in traj ]
    except FileNotFoundError:
        print('[WARNING] file \'{}\' does not exist, it will be skipped'.format(est_path))

    # write trajectory in KITTI format
    with open(out_path, 'w') as f:
        f.writelines(traj_str)

    print(' done! Trajectory written at {}'.format(out_path))

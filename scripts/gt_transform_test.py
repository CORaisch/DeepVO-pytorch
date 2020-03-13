#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

# NOTE this file shows that the transformation from abs matrix to rel euler poses is not working properly and should be reimplemented in original code

# FIXME compare to test.py: why getting one additional pose out?
# FIXME save answer in 12 DoF format like original KITTI poses

# bultins
import os, glob, argparse
from pathlib import Path
# project dependencies
from data_helper import get_data_info, ImageSequenceDataset
from utils import eulerAnglesToRotationMatrix, normalize_angle_delta
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

    def __getitem__(self, index):
        raw_groundtruth = np.hsplit(self.groundtruth_arr[index], np.array([6]))
        groundtruth_sequence = raw_groundtruth[0]
        groundtruth_rotation = raw_groundtruth[1][0].reshape((3, 3)).T
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)
        groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0]
        for gt_seq in groundtruth_sequence[1:]:
            location = torch.FloatTensor(groundtruth_rotation.dot(gt_seq[3:].numpy()))
            gt_seq[3:] = location[:]
        groundtruth_sequence[2:] = groundtruth_sequence[2:] - groundtruth_sequence[1:-1]
        for gt_seq in groundtruth_sequence[1:]:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])
        return groundtruth_sequence

    def __len__(self):
        return len(self.data_info.index)

# parse arguments
argparser = argparse.ArgumentParser(description="transforms DVO poses as used for training back to 12 DoF absolute poses. Will be used to validate ds transformation.")
argparser.add_argument('out', type=str, help="path where results will be saved")
argparser.add_argument('dataset', type=str, help="dataset base directory")
argparser.add_argument('sequences', type=str, nargs='+', help="video indices to transform")
argparser.add_argument('--batch_size', '-bs', type=int, default=8, help="batch size for transforming (default: 8)")
argparser.add_argument('--only_yaw', '-only_yaw', action='store_true', help="if set only yaw angle will be used")
args = argparser.parse_args()

if __name__ == '__main__':
    # set dataset to test on
    image_dir = os.path.join(args.dataset, 'images')
    pose_dir = os.path.join(args.dataset, 'poses_gt')

    # prepare directory structure
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # prepare dataset
    n_workers = 1
    seq_len = 6
    overlap = seq_len - 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))

    # loop over sequences
    for seq in args.sequences:
        # create ds to iterate
        df = get_data_info(image_dir, pose_dir, folder_list=[seq], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        df = df.loc[df.seq_len == seq_len]  # drop last
        dataset = PoseSequenceDataset(df)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers)

        # loop over sequence
        answer = [[0.0]*6, ]
        n_batch = len(dataloader)
        for i, batch in enumerate(dataloader):
            print('{} / {}'.format(i, n_batch), end='\r', flush=True)
            batch_pose_sequences = batch # take labels only

            batch_pose_sequences = batch_pose_sequences.data.cpu().numpy()
            if i == 0:
                for pose in batch_pose_sequences[0]:
                    # use all predicted pose in the first prediction
                    for i in range(len(pose)):
                        # Convert predicted relative pose to absolute pose by adding last pose
                        pose[i] += answer[-1][i]
                    answer.append(pose.tolist())
                batch_pose_sequences = batch_pose_sequences[1:]

            # Transform from relative to absolute
            for predict_pose_seq in batch_pose_sequences:
                if args.only_yaw:
                    ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0])
                else:
                    ang = eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]])
                location = ang.dot(predict_pose_seq[-1][3:])
                predict_pose_seq[-1][3:] = location[:]

                # use only last predicted pose in the following prediction
                last_pose = predict_pose_seq[-1]
                for i in range(len(last_pose)):
                    last_pose[i] += answer[-1][i]
                # normalize angle to -Pi...Pi over y axis
                last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                answer.append(last_pose.tolist())

        print('len(answer):', len(answer))
        print('expect len:', len(glob.glob(os.path.join(image_dir, seq, '*.png'))))

        # FIXME since I still get one pose too much I will drop the last one as hack solution until I solve the problem
        answer = answer[:-1] if len(answer) == len(glob.glob(os.path.join(image_dir, seq, '*.png'))) + 1 else answer

        # FIXME save answer in 12 DoF format like original KITTI poses
        # save answer
        with open(os.path.join(args.out, 'out_{}.txt'.format(seq)), 'w') as f:
            # for pose in answer:
            for pose in answer:
                if type(pose) == list:
                    f.write(', '.join([str(p) for p in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')

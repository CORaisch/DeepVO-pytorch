# builtins
import glob, os, time, argparse
from pathlib import Path
from PIL import Image
# project dependencies
from params import par
from model import DeepVO
from helper import eulerAnglesToRotationMatrix
from data_helper import get_data_info, ImageSequenceDataset
# external dependencies
import numpy as np
import torch
from torch.utils.data import DataLoader

# parse passed arguments
argparser = argparse.ArgumentParser(description="DeepVO Testing")
argparser.add_argument('model', type=str, help="path to trained model")
argparser.add_argument('out', type=str, help="path where estimates will be saved")
argparser.add_argument('--dataset_dir', '-ds', type=str, default=None, help="directory of dataset, if not set it will be read from params")
argparser.add_argument('--sequences', '-seq', type=str, default=None, nargs='+', help="list of video sequences (indices) used for preprocessing, if not set it will be read from params")
args = argparser.parse_args()

if __name__ == '__main__':

    # Specify dataset to test on
    if args.dataset_dir:
        par.data_dir = args.dataset_dir
        par.image_dir = os.path.join(par.data_dir, 'images')
        par.pose_dir = os.path.join(par.data_dir, 'poses_gt')

    # Specify video sequences to test on
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = par.train_seq + list(set(par.valid_seq) - set(par.train_seq)) # NOTE train_video ∪ valid_video, i.e. removing duplicates if exist

    # Prepare directory structure
    load_model_path = args.model
    save_dir = args.out
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        M_deepvo = M_deepvo.cuda()
        M_deepvo.load_state_dict(torch.load(load_model_path))
    else:
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location={'cuda:0': 'cpu'}))
    print('Load model from: ', load_model_path)

    # Prepare dataset
    n_workers = 1
    seq_len = int((par.seq_len[0]+par.seq_len[1])/2)
    overlap = seq_len - 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))
    batch_size = par.batch_size

    # Test loop
    for test_seq in sequences:
        # make dataloader
        df = get_data_info(folder_list=[test_seq], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        df = df.loc[df.seq_len == seq_len]  # drop last
        dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        # load gt poses
        gt_pose = np.load(os.path.join(par.pose_dir, '{}.npy'.format(test_seq))) # (n_images, 6)

        # Predict
        M_deepvo.eval()
        has_predict = False
        answer = [[0.0]*6, ]
        st_t = time.time()
        n_batch = len(dataloader)

        for i, batch in enumerate(dataloader):
            print('{} / {}'.format(i, n_batch), end='\r', flush=True)
            _, x, y = batch
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            batch_predict_pose = M_deepvo.forward(x)

            batch_predict_pose = batch_predict_pose.data.cpu().numpy()
            if i == 0:
                for pose in batch_predict_pose[0]:
                    # use all predicted pose in the first prediction
                    for i in range(len(pose)):
                        # Convert predicted relative pose to absolute pose by adding last pose
                        pose[i] += answer[-1][i]
                    answer.append(pose.tolist())
                batch_predict_pose = batch_predict_pose[1:]

            # Transform from relative to absolute
            for predict_pose_seq in batch_predict_pose:
                # predict_pose_seq[1:] = predict_pose_seq[1:] + predict_pose_seq[0:-1]
                ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0]) #eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]])
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
        print('expect len:', len(glob.glob(os.path.join(par.image_dir, test_seq, '*.png'))))
        print('Predict use {} sec'.format(time.time() - st_t))

        # Save answer
        with open(os.path.join(save_dir, 'out_{}.txt'.format(test_seq)), 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(', '.join([str(p) for p in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')

        # Calculate loss
        loss = 0
        for t in range(len(gt_pose)):
            angle_loss = np.sum((answer[t][:3] - gt_pose[t,:3]) ** 2)
            translation_loss = np.sum((answer[t][3:] - gt_pose[t,3:6]) ** 2)
            loss = (100 * angle_loss + translation_loss)
        loss /= len(gt_pose)
        print('Loss = ', loss)
        print('='*50)

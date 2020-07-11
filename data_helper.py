# builtins
import os, glob, time
from PIL import Image
from random import shuffle
# project dependencies
from params import par
from utils import normalize_angle_delta
# external dependencies
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms

def get_data_info(image_dir, pose_dir, folder_list, seq_len_range, overlap, sample_times=1, max_step=1, pad_y=False, shuffle=False, sort=True):
    # check inputs
    assert overlap < min(seq_len_range)
    assert max_step > 0
    # subsequene each sequence
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        start_t = time.time()
        poses = np.load(os.path.join(pose_dir, '{}.npy'.format(folder))) # (n_images, 15)
        fpaths = glob.glob(os.path.join(image_dir, folder, '*.png'))
        fpaths.sort()
        # Random segment to sequences with diff lengths
        n_frames = len(fpaths)
        min_len, max_len = seq_len_range[0], seq_len_range[1]
        for i in range(sample_times):
            start = 0
            while True:
                n = np.random.random_integers(min_len, max_len)
                s = np.random.random_integers(1, max_step)
                if start + n*s <= n_frames:
                    x_seg = fpaths[start:start+n*s:s]
                    X_path.append(x_seg)
                    if not pad_y:
                        Y.append(poses[start:start+n*s:s])
                    else:
                        pad_zero = np.zeros((max_len-n, 15))
                        padded = np.concatenate((poses[start:start+n*s:s], pad_zero))
                        Y.append(padded.tolist())
                else:
                    print('Last %d frames is not used' %(start+(n*s)-n_frames))
                    break
                start += s * (n - overlap)
                X_len.append(len(x_seg))
        print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    # Sort dataframe by seq_len
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df

# TODO test function
def get_partition_data_info(image_dir, pose_dir, partition, folder_list, seq_len_range, overlap, sample_times=1, max_step=1, pad_y=False, sort=True):
    # partition must be in ]0.0,1.0[
    assert partition > 0.0 and partition < 1.0
    # get total dataframe
    total_df = get_data_info(
        image_dir, pose_dir, folder_list, seq_len_range, overlap, sample_times=sample_times, max_step=max_step, pad_y=pad_y, shuffle=False, sort=False)
    # split total_df into two given the value of partition
    msk = np.random.rand(total_df.shape[0]) < partition
    train_df = total_df[msk]
    valid_df = total_df[~msk]
    # sort dataframe by seq_len
    if sort:
        train_df = train_df.sort_values(by=['seq_len'], ascending=False)
        valid_df = valid_df.sort_values(by=['seq_len'], ascending=False)
    return train_df, valid_df

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        # NOTE so far the order of 'list_batch_indexes' is like: [all batches with seq_len x, all batches with seq_len x-1, ...],
        #      maybe we should finally shuffle 'list_batch_indexes' to prevent correlations according to sequence length
        shuffle(list_batch_indexes) # FIXME test with this line enabled
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


# NOTE is sequence_len_list neccesarry ? seems like its rudimentary
# FIXME label transformation is not working correctly -> fix it
class ImageSequenceDataset(Dataset):

    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        if par.grayscale:
            transform_ops.append(transforms.Grayscale(num_output_channels=3))
        if par.laplace_preprocessing:
            transform_ops.append(transforms.Lambda(preprocess_laplace))
        transform_ops.append(transforms.ToTensor())
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)

        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
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
        seq_rel = [ self._inv(seq_abs[0]) * seq_abs[i+1] for i in range(seq_len-1) ]
        seq_gt  = [ np.concatenate((self._to_euler(T[:3,:3]), np.asarray(T[:3,3])), axis=None) for T in seq_rel ]

        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path).convert('RGB')
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            if not par.laplace_preprocessing: # NOTE only normalize if NO laplace preprocessing applied
                img_as_tensor = self.normalizer(img_as_tensor)

            # # beg DEBUG
            # show_tensor_image(img_as_tensor, title="image after transformer")
            # # end DEBUG

            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        # NOTE 'groundtruth_sequence' must contain as many poses as 'image_sequence' has images. The first pose will be omited before loss computation.
        return (sequence_len, image_sequence, torch.FloatTensor(seq_gt))

    def __len__(self):
        return len(self.data_info.index)

def preprocess_laplace(im):
    import cv2 as cv
    rad = 3
    im  = np.array(im)
    s   = im.shape
    out = np.zeros((s[0],s[1],3), dtype=im.dtype)
    for c in range(3):
        layer = im[:,:,c]
        layer = cv.medianBlur(layer, rad)
        layer = cv.Laplacian(layer, cv.CV_16S, ksize=rad)
        layer = cv.convertScaleAbs(layer)
        out[:,:,c] = layer
    return Image.fromarray(out)

def show_tensor_image(image, title="", colormap=None):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    sample_image = TF.to_pil_image(image)
    plt.imshow(sample_image, cmap=colormap)
    plt.title(title)
    plt.show()

def show_pil_image(image, title="", colormap=None):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    plt.imshow(image, cmap=colormap)
    plt.title(title)
    plt.show()

# Example of usage
if __name__ == '__main__':
    start_t = time.time()
    # Gernerate info dataframe
    overlap = 1
    sample_times = 1
    folder_list = ['00']
    seq_len_range = [5, 7]
    df = get_data_info(folder_list, seq_len_range, overlap, sample_times)
    print('Elapsed Time (get_data_info): {} sec'.format(time.time()-start_t))
    # Customized Dataset, Sampler
    n_workers = 4
    resize_mode = 'crop'
    new_size = (150, 600)
    img_mean = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    dataset = ImageSequenceDataset(df, resize_mode, new_size, img_mean)
    sorted_sampler = SortedRandomBatchSampler(df, batch_size=4, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=sorted_sampler, num_workers=n_workers)
    print('Elapsed Time (dataloader): {} sec'.format(time.time()-start_t))

    for batch in dataloader:
        s, x, y = batch
        print('='*50)
        print('len:{}\nx:{}\ny:{}'.format(s, x.shape, y.shape))

    print('Elapsed Time: {} sec'.format(time.time()-start_t))
    print('Number of workers = ', n_workers)


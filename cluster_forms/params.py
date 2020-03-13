import os
from utils import Singleton

@Singleton
class Parameters():
    def __init__(self):

        ## Dataset

        # subsequence generation
        self.seq_len = (5, 7) # set min and max length of subsequences for RNN training
        self.sample_times = 3 # when generating subsequences loop will iterate 'sample_times' times over the entire DS
        self.max_step = 1 # if set > 1 for all subsequence each rand(1,max_step) will be skipped, this way more distances are simulated

        # preprocessing
        self.grayscale = False # specifiy if grayscale images should be used for training/testing
        self.laplace_preprocessing = False # enable laplace preprocessing NOTE instead of normalizing the inputs with mean and stdev, images will be laplace filtered
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        # KITTI
        # self.img_w = 608 # original size is about 1226
        # self.img_h = 184 # original size is about 370
        # CARLA
        self.img_w = 400 # original size is about 800
        self.img_h = 300 # original size is about 600
        # means and std for CARLA RGB
        self.img_means = (-0.08028274693997746, -0.09296754998591701, -0.1406916200942229)
        self.img_stds = (0.22807312410203096, 0.2269092573945764, 0.23238783542793442)
        # means and std for KITTI RGB
        # self.img_means = (-0.15188777921929586, -0.13248322798940773, -0.13754771375070027)
        # self.img_stds = (0.31139055400107185, 0.31645172811357664, 0.3233846633856191)
        # means and stdev for KITTI GRAY
        # self.img_means = () # FIXME add values
        # self.img_stds = () # FIXME add values
        # means and stdev for CARLA GRAY
        # self.img_means = () # FIXME add values
        # self.img_stds = () # FIXME add values
        self.minus_point_5 = True # pixels will be shifted in range [-0.5, 0.5] (required when using pretrained flownet weights)

        # Model
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0   # 0: no dropout
        self.clip = None
        self.batch_norm = True

        # Training
        self.pin_mem = True
        self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
                    # Choice:
                    # {'opt': 'Adagrad', 'lr': 0.001}
                    # {'opt': 'Adam'}
                    # {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}

par = Parameters.instance()

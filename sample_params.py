import os

class Parameters():
    def __init__(self):
        ## Datasets
        self.n_processors = 4
        # path to DS 1
        self.data_dir =  '/home/claudio/Datasets/CARLA/DeepVO/'
        self.image_dir = os.path.join(self.data_dir, 'images/')
        self.pose_dir = os.path.join(self.data_dir, 'poses_gt/')

        # video sequences
        # KITTI
        # self.train_seq = ['00', '01', '02', '05', '08', '09']
        # self.valid_seq = ['04', '06', '07', '10']
        # CARLA
        self.train_seq = ['00', '01', '03', '05', '06', '07', '09']
        self.valid_seq = ['02', '04', '08']
        self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8

        # DS generation
        self.seq_len = (5, 7) # set min and max length of subsequences for RNN training
        self.sample_times = 3 # when generating subsequences loop will iterate 'sample_times' times over the entire DS

        ## Data Preprocessing
        self.grayscale = False # specifiy if grayscale images should be used for training/testing
        self.laplace_preprocessing = False # enable laplace preprocessing NOTE instead of normalizing the inputs with mean and std images will be laplace filtered
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        # KITTI
        # self.img_w = 608 # original size is about 1226
        # self.img_h = 184 # original size is about 370
        # CARLA
        self.img_w = 400 # original size is about 800
        self.img_h = 300 # original size is about 600
        # means and std for KITTI RGB
        # self.img_means = (-0.15188777921929586, -0.13248322798940773, -0.13754771375070027)
        # self.img_stds = (0.31139055400107185, 0.31645172811357664, 0.3233846633856191)
        # means and std for KITTI GRAY
        # self.img_means = () # FIXME add values
        # self.img_stds = () # FIXME add values
        # means and std for CARLA RGB
        self.img_means = (0.028457542237100963, -0.006877451946295531, -0.002785168301653312)
        self.img_stds = (0.2152724173699555, 0.229336494250576, 0.2559454319056272)
        self.minus_point_5 = True # pixels will be shifted in range [-0.5, 0.5] (required when using pretrained flownet weights)

        # Model
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0   # 0: no dropout
        self.clip = None
        self.batch_norm = True

        # Training
        self.epochs = 400
        self.batch_size = 8
        self.pin_mem = True
        self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
                    # Choice:
                    # {'opt': 'Adagrad', 'lr': 0.001}
                    # {'opt': 'Adam'}
                    # {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}

        # Pretrained FlowNet weights
        self.pretrained_flownet = 'pretrained/flownets_EPE1.951.pth.tar'
                                # Choice:
                                # None
                                # './pretrained/flownets_bn_EPE2.459.pth.tar'
                                # './pretrained/flownets_EPE1.951.pth.tar'

par = Parameters()


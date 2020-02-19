import os

class Parameters():
    def __init__(self, unpack_to = None):
        self.n_processors = 4
        # Path
        self.data_dir =  '/home/claudio/Datasets/CARLA/DeepVO/'
        self.image_dir = os.path.join(self.data_dir, 'images/')
        self.pose_dir = os.path.join(self.data_dir, 'poses_gt/')

        # KITTI
        # self.train_video = ['00', '01', '02', '05', '08', '09']
        # self.valid_video = ['04', '06', '07', '10']
        # CARLA
        self.train_video = ['00', '01', '03', '05', '06', '07', '09']
        self.valid_video = ['02', '04', '08']
        self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8


        # Data Preprocessing
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
        self.minus_point_5 = True

        self.seq_len = (5, 7)
        self.sample_times = 3

        # Data info path
        self.train_data_info_path = 'datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
        self.valid_data_info_path = 'datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)


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

        # Pretrain, Resume training
        # self.pretrained_flownet = None
        self.pretrained_flownet = 'pretrained/flownets_EPE1.951.pth.tar'
                                # Choice:
                                # None
                                # './pretrained/flownets_bn_EPE2.459.pth.tar'
                                # './pretrained/flownets_EPE1.951.pth.tar'
        # self.resume = True  # resume training
        self.resume = False
        self.resume_t_or_v = '.train'

        self.experiment_name = '/sample_name/'
        self.load_model_path = 'models{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(self.experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.load_optimizer_path = 'models{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(self.experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)

        self.record_path = 'records{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(self.experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
        self.save_model_path = 'models{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(self.experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
        self.save_optimzer_path = 'models{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(self.experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))

        self.results_dir = 'results{}'.format(self.experiment_name)


        if not os.path.isdir(os.path.dirname(self.record_path)):
            os.makedirs(os.path.dirname(self.record_path))
        if not os.path.isdir(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
            os.makedirs(os.path.dirname(self.save_optimzer_path))
        if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
            os.makedirs(os.path.dirname(self.train_data_info_path))

    def set_remote_dir(self, remote_dir):
        if remote_dir:
            self.data_dir = remote_dir
            self.image_dir = os.path.join(self.data_dir, 'images/')
            self.pose_dir = os.path.join(self.data_dir, 'poses_gt/')

    def set_home_dir(self, home_dir):
        if home_dir:
            self.train_data_info_path = os.path.join(home_dir, self.train_data_info_path)
            self.valid_data_info_path = os.path.join(home_dir, self.valid_data_info_path)
            if self.pretrained_flownet:
                self.pretrained_flownet = os.path.join(home_dir, self.pretrained_flownet)
            self.load_model_path = os.path.join(home_dir, self.load_model_path)
            self.load_optimizer_path = os.path.join(home_dir, self.load_optimizer_path)
            self.record_path = os.path.join(home_dir, self.record_path)
            self.save_model_path = os.path.join(home_dir, self.save_model_path)
            self.save_optimzer_path = os.path.join(home_dir, self.save_optimzer_path)
            self.results_dir = os.path.join(home_dir, self.results_dir)

    def set_resume(self, val):
        self.resume = val

par = Parameters()


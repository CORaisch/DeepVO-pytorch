import os

class Parameters():
    def __init__(self, unpack_to = None):
        self.n_processors = 4
        # Path
        self.data_dir =  '/media/claudio/1AC5-C2D4/Datasets/KITTI/DeepVO-pytorch/'
        self.image_dir = os.path.join(self.data_dir, 'images/')
        self.pose_dir = os.path.join(self.data_dir, 'poses_gt/')

        self.train_video = ['00', '01', '02', '05', '08', '09']
        self.valid_video = ['04', '06', '07', '10']
        self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8


        # Data Preprocessing
        self.grayscale = False # specifiy if grayscale images should be used for training/testing
        self.laplace_preprocessing = False # enable laplace preprocessing NOTE instead of normalizing the inputs with mean and std images will be laplace filtered
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        self.img_w = 608   # original size is about 1226
        self.img_h = 184   # original size is about 370
        # means and std for kitti rgb image sequence [00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10]
        self.img_means = (-0.15116102640573548, -0.1322411015338543, -0.13887598313286317)
        self.img_stds = (0.31308950448998596, 0.3176070324487968, 0.3232656266278995)
        # means and std for kitti grayscale image sequence [00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10]
        # self.img_means = (-0.14046805191159092, -0.14046805191159092, -0.14046805191159092)
        # self.img_stds = (0.3128824310845576, 0.3128824310845576, 0.3128824310845576)
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
        self.pretrained_flownet = None
                                # Choice:
                                # None
                                # './pretrained/flownets_bn_EPE2.459.pth.tar'
                                # './pretrained/flownets_EPE1.951.pth.tar'
        # self.resume = True  # resume training
        self.resume = False
        self.resume_t_or_v = '.train'

        experiment_name = '/rgb_scratch_ep400/'
        self.load_model_path = 'models{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.load_optimizer_path = 'models{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)

        self.record_path = 'records{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
        self.save_model_path = 'models{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
        self.save_optimzer_path = 'models{}t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(experiment_name, ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))

        self.results_dir = 'results{}'.format(experiment_name)


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


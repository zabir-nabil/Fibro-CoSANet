# author: github/zabir-nabil

class HyperP:
    def __init__(self, model_type):
        # hyperparameters
        if model_type == "slope_train":
            self.seed = 1997
            self.data_folder = '..' # one level up
            self.ct_tab_feature_csv = 'train_data_ct_tab.csv' # some extra features
            self.strip_ct = .15 # strip this amount of ct slices before randomly choosing
            self.n_tab = 5 # number of tabular features used

            self.cnn_dim = 32 # compressed cnn feature dim

            self.fc_dim = 16

            # select which models to train
            self.train_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'] 

            self.gpu_index = 0
            self.num_workers = 0 # 0 for bug fix/docker
            self.results_dir = "results_slopes"
            self.nfold = 5
            self.n_epochs = 40
            self.batch_size = 16
            self.final_lr = 0.0002
        elif model_type == "slope_test":
            pass
        elif model_type == "qreg_train":
            self.seed = 1997
            self.data_folder = '..'
            self.ct_tab_feature_csv = 'train_data_ct_tab.csv' # some extra features
            self.strip_ct = .15 # strip this amount of ct slices before randomly choosing
            self.n_tab = 7 # number of tabular features used

            # select which models to train
            self.train_models = ['resnet18' , 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'] 

            self.gpu_index = 0
            self.results_dir = "results_qreg"
            self.nfold = 5
            self.n_epochs = 40
            self.batch_size = 8
            self.final_lr = 0.0002
            self.loss_weight = 0.7
            self.dummy_training = False
            self.dummy_train_rows = 400
        elif model_type == "attn_train":
            # ablation study
            self.seed = 1997
            self.data_folder = '..' # .. one level up
            self.ct_tab_feature_csv = 'train_data_ct_tab.csv' # some extra features
            self.strip_ct = .15 # strip this amount of ct slices before randomly choosing
            self.n_tab = 5 # number of tabular features used

            # self.cnn_dim = 32 # compressed cnn feature dim

            self.fc_dim = [16, 32]

            # select which models to train
            self.train_models = ['efnb2_attn'] 

            self.gpu_index = 0
            self.num_workers = 0 # 0 for bug fix/docker
            self.results_dir = "results_attn"
            self.nfold = 5
            self.n_epochs = 40
            self.batch_size = 10
            self.final_lr = 0.0002

            self.attn_filters = [32, 64, 128] # attn_filters and cnn_dim should be same

            self.n_attn_layers = [1, 2, 3]

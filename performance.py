# author: github/zabir-nabil

# relevant imports

import os
import cv2

import pydicom
import pandas as pd
import numpy as np 
# import tensorflow as tf 
# import matplotlib.pyplot as plt 

# torch dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import random
from tqdm import tqdm 

# k-fold
from sklearn.model_selection import KFold

# hyperparam object

from config import HyperP
from time import process_time 
from ptflops import get_model_complexity_info

hyp = HyperP(model_type = "attn_train_best_config") # slope prediction

# seed
seed = hyp.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# path 

root_path = hyp.data_folder # ../input/osic-pulmonary-fibrosis-progression

train = pd.read_csv(f'{root_path}/train.csv') 
train_vol = pd.read_csv(f'{hyp.ct_tab_feature_csv}') 


train['Volume'] = 2000.

for i in range(len(train)):
    pid = train.iloc[i]['Patient']
    try:
        train.at[i, 'Volume'] = train_vol[train_vol['Patient']==pid].iloc[0]['Volume']
    except:
        print('bug at volume')


# tabular feature generation

def get_tab(df):
    vector = [(df.Age.values[0] - train.Age.values.mean()) /  train.Age.values.std()] # df.Age.values[0].mean(), df.Age.values[0].std()
    
    if df.Sex.values[0] == 'Male':
        vector.append(0)
    else:
        vector.append(1)
    
    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0]) # this is useless
        
    vector.append((df.Volume.values[0] - train.Volume.values.mean()) /  train.Volume.values.std())
    return np.array(vector) 


A = {} # the slopes
TAB = {} # tabular features
P = [] # patient IDs

for i, p in tqdm(enumerate(train.Patient.unique())):
    sub = train.loc[train.Patient == p, :] 
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    
    a, _ = np.linalg.lstsq(c, fvc)[0] # we calculate the best slope with least square
    
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    
    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)





class OSICData(Dataset):
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    def __init__(self, keys, a, tab):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.tab = tab
        
        self.train_data = {}
        for p in train.Patient.values:
            p_n = len(os.listdir(f'{root_path}/train/{p}/'))
            self.train_data[p] = os.listdir(f'{root_path}/train/{p}/')[int( hyp.strip_ct * p_n):-int( hyp.strip_ct * p_n)] # removing first and last 15% slices
    
    
    def __len__(self):
        return len(self.keys)

    def get_img(self, path):
        d = pydicom.dcmread(path)
        return cv2.resize(d.pixel_array / 2**11, (512, 512)) # maybe bug in resize
    
    def __getitem__(self, idx):
        x = []
        a, tab = [], [] 
        k = self.keys[idx] # instead of random id send a specific id
        # np.random.choice(self.keys, 1)[0]

        try:
            i = np.random.choice(self.train_data[k], size=1)[0]
            # print(i)
            cp = f'{root_path}/train/{k}/{i}'
            #print(cp)
            img = self.get_img(cp)
            
            x.append(img)
            a.append(self.a[k])
            tab.append(self.tab[k])
        except:
            print('failed')
            print(k, i)
       
        x, a, tab = torch.tensor(x, dtype=torch.float32), torch.tensor(a, dtype=torch.float32), torch.tensor(tab, dtype=torch.float32)
        tab = torch.squeeze(tab, axis=0)
        return [x, tab] , a, k # k for patient id


from torchvision import models
from torch import nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

class Identity(nn.Module):
    # credit: ptrblck
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.rand(1)) # random initialization

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out # , attention


# only based on best config of b2
class TabCT(nn.Module):
    def __init__(self, cnn, attn_filters, fc_dim, n_attn_layers):
        super(TabCT, self).__init__()
        
        # CT features
        cnn_dict = {'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50,
                   'resnet101': models.resnet101, 'resnet152': models.resnet152, 'resnext50': models.resnext50_32x4d,
                   'resnext101': models.resnext101_32x8d}
        
        # feature dim
        self.out_dict = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048, 'resnet152': 2048,
                         'resnext50': 2048, 'resnext101': 2048, "efnb0": 1280, "efnb1": 1280, "efnb2": 1408, 
                          "efnb3": 1536, "efnb4": 1792, "efnb5": 2048, "efnb6": 2304, "efnb7": 2560}
        
        self.n_tab = hyp.n_tab # n tabular features
        self.attn_filters = attn_filters
        self.fc_dim = fc_dim
        self.n_attn_layers = n_attn_layers
        
        # efficient net b2 base
        if cnn in cnn_dict.keys():
            self.ct_cnn = cnn_dict[cnn](pretrained = True)
            
            # make single channel
            self.ct_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            self.ct_cnn.avgpool = nn.Conv2d(self.out_dict[cnn], self.attn_filters, kernel_size=(1, 1), bias=False)
            self.ct_cnn.fc = nn.Identity()
            
            # 1 self attn layer [stacked]
            self.attn = nn.ModuleList()
            
            for _ in range(self.n_attn_layers):
                self.attn.append(Self_Attn(self.attn_filters))
        elif 'efn' in cnn:
            if 'b0' in cnn:
                self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b0')
                self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 32, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            elif 'b1' in cnn:
                self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b1')
                self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 32, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)            
            elif 'b2' in cnn:
                self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b2')
                self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 32, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)     
            elif 'b3' in cnn:
                self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b3')
                self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 40, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)     
            elif 'b4' in cnn:
                self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b4')
                self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 48, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)               
            # replace avg_pool layer
            # 1408 is the number of filters in last conv
            self.ct_cnn._avg_pooling = Conv2dStaticSamePadding(self.out_dict[ cnn.split('_')[0] ], self.attn_filters, kernel_size = (1,1), stride = (1,1), 
                                                 bias = False, image_size = 512)
            self.ct_cnn._dropout = nn.Identity()
            self.ct_cnn._fc = nn.Identity()
            self.ct_cnn._swish = nn.Identity()
            
            # 1 self attn layer [stacked]
            self.attn = nn.ModuleList()
            
            for _ in range(self.n_attn_layers):
                self.attn.append(Self_Attn(self.attn_filters))

                
        else:
            raise ValueError("cnn not recognized")
            
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.dropout = nn.Dropout(p=0.25)
        
        self.fc_inter = nn.Linear(self.attn_filters + self.n_tab, self.fc_dim)

        self.fc = nn.Linear(self.fc_dim, 1)
        
    def forward(self, x_ct, x_tab):
        print(x_ct.shape)
        ct_f = self.ct_cnn(x_ct).view(-1, self.attn_filters, 16, 16) # ct features
        #print(ct_f.shape)
        
        for ii in range(len(self.attn)):
            ct_f = self.attn[ii](ct_f)
        #print(ct_f.shape)
        ct_f = self.avgpool(ct_f).view(-1, self.attn_filters)
        #print(ct_f.shape)
        # print(x_tab.shape)
        
        # concatenate
        x = torch.cat((ct_f, x_tab), -1) # concat on last axis
        
        # dropout
        if self.training:
             x = self.dropout(x)
                
        x = self.fc_inter(x)

        x = self.fc(x)
        
        return x

from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error





# hyperparams

result_dir = "perf_results"

# training only resnet models on gpu 0
train_models = hyp.train_models 

# 'resnext101' -> seems too heavy for 1080
# 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'

# device
gpu = torch.device(f"cuda:{hyp.gpu_index}" if torch.cuda.is_available() else "cpu")



# removing noisy data
P = [p for p in P if p not in ['ID00011637202177653955184', 'ID00052637202186188008618']]


# mac and param calc
def prepare_input(resolution):
    x1 = torch.FloatTensor(1, 1, 512, 512).to(torch.device("cuda"))
    x2 = torch.FloatTensor(1, 5).to(torch.device("cuda"))
    return {"x_ct" : x1, "x_tab" : x2}

for model in train_models:
    for fd in hyp.fc_dim:
        for af in hyp.attn_filters:
            for nal in hyp.n_attn_layers:
                log = open(f"{result_dir}/{model}_fd_{fd}_af_{af}_nal_{nal}.txt", "a+")
                kfold =KFold(n_splits=2)
                
                ifold = 0
                for train_index, test_index in kfold.split(P):  
                    # print(train_index, test_index) 

                    p_train = np.array(P)[train_index] 
                    p_test = np.array(P)[test_index] 
                    
                    osic_train = OSICData(p_train, A, TAB)
                    train_loader = DataLoader(osic_train, batch_size=1, shuffle=True, num_workers=hyp.num_workers)

                    osic_val = OSICData(p_test, A, TAB)
                    val_loader = DataLoader(osic_val, batch_size=1, shuffle=True, num_workers=hyp.num_workers)
                    
                
                    tabct = TabCT(cnn = model, fc_dim = fd, attn_filters = af, n_attn_layers = nal).to(gpu) # net

                    macs, params = get_model_complexity_info(tabct, input_res = (1, 512, 512), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=True, verbose=True)

                    print(f"creating {model} with {fd} feature_dim, {af} attn_filters, and {nal} n_attn_layers")

                    print(f"macs : {macs} params : {params}")
                    log.write(f"macs : {macs} params : {params}\n")

                    g_count = 0
                    t1 = process_time()
                    for epoch in range(5):  # loop over the dataset multiple times

                        tabct.eval()
                        for i, data in tqdm(enumerate(val_loader, 0)):

                            [x, t], a, pid = data

                            x = x.to(gpu)
                            t = t.to(gpu)
                            a = a.to(gpu)

                            # forward
                            outputs = tabct(x, t)

                            g_count += 1

                            if g_count >= 10:
                                break

                        if g_count >= 10:
                            break
                    t2 = process_time()
                    print(f"infer: {(t2-t1)/g_count}")
                    log.write(f"infer: {(t2-t1)/g_count}\n")
                    # destroy model
                    del tabct
                    torch.cuda.empty_cache()
                    break




# ref: https://www.kaggle.com/miklgr500/linear-decay-based-on-resnet-cnn
# https://pytorch.org/docs/stable/index.html
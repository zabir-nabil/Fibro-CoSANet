# author: github/zabir-nabil

# relevant imports

import os
import cv2

import pydicom
import pandas as pd
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

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

hyp = HyperP(model_type = "slope_train") # slope prediction

# seed
seed = hyp.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# path 

root_path = hyp.data_folder # ../input/osic-pulmonary-fibrosis-progression

train = pd.read_csv(f'{root_path}/train.csv') 
train_vol = pd.read_csv(f'{hyp.ct_tab_feature_csv}') 


# getting base week for patient
def get_baseline_week(data):
    df = data.copy()
    df['Weeks'] = df['Weeks'].astype(int)
    df['min_week'] = df.groupby('Patient')['Weeks'].transform('min')
    df['baseline_week'] = df['Weeks'] - df['min_week']
    return df

# getting FVC for base week and setting it as base_FVC of patient
def get_base_FVC(data):
    df = data.copy()
    base = df.loc[df.Weeks == df.min_week][['Patient','FVC']].copy()
    base.columns = ['Patient','base_FVC']
    
    base['nb']=1
    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
    
    base = base[base.nb==1]
    base.drop('nb',axis =1,inplace=True)
    df = df.merge(base,on="Patient",how='left')
    df.drop(['min_week'], axis = 1)
    return df 

# getting Percent for base week and setting it as base_Percent of patient
def get_base_Percent(data):
    df = data.copy()
    
    return df 


train['Volume'] = 2000.

for i in range(len(train)):
    pid = train.iloc[i]['Patient']
    try:
        train.at[i, 'Volume'] = train_vol[train_vol['Patient']==pid].iloc[0]['Volume']
    except:
        print('bug at volume')


# encoding temporal info
train_data = train
train_data.drop_duplicates(keep=False,inplace=True,subset=['Patient','Weeks'])
train_data = get_baseline_week(train_data)
train_data = get_base_FVC(train_data)


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



def get_img(path):
    d = pydicom.dcmread(path)
    return cv2.resize(d.pixel_array / 2**11, (512, 512))

# torch dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class OSICData(Dataset):
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    def __init__(self, keys, train_df):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.train_df = train_df
        
        self.train_data = {}
        for p in train.Patient.values:
            p_n = len(os.listdir(f'{root_path}/train/{p}/'))
            self.train_data[p] = os.listdir(f'{root_path}/train/{p}/')[int( hyp.strip_ct * p_n):-int( hyp.strip_ct * p_n)] # removing first and last 15% slices
    
    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, idx):
        x = []
        tab = []
        all_features = list(self.train_df.iloc[idx])
        pid = all_features[0]
        fvc = []
        fvc.append(all_features[2])
        try:
            i = np.random.choice(self.train_data[pid], size=1)[0]
            img = get_img(f'{root_path}/train/{pid}/{i}')
            x.append(img)
            tab.append(all_features[1:5])
        except Exception as e:
            print(e)
            print('error')
        x, tab, fvc = torch.tensor(x, dtype=torch.float32), torch.tensor(tab, dtype=torch.float32), torch.tensor(fvc, dtype=torch.float32)
        tab = torch.squeeze(tab, axis=0)
        return [x, tab], fvc


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
    
class TabCT(nn.Module):
    def __init__(self, cnn):
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
        
        # hashtable features
        self.cnn_features = {} # a map pid to feature
        
        # efficient net b0 to b7
        
        if cnn in cnn_dict.keys(): # resnet or resnext
            self.ct_cnn = cnn_dict[cnn](pretrained = True)
            
            # make single channel
            self.ct_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            # remove the fc layer/ add a simple linear layer
            self.ct_cnn.fc = nn.Linear(self.out_dict[cnn], 64)   # mapped to 64 dimensions, Identity()
            
        elif cnn == "efnb0":
            self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b0')
            self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 32, kernel_size = (3,3), stride = (2,2), 
                                                             bias = False, image_size = 512)
            self.ct_cnn._fc = nn.Linear(self.out_dict[cnn], 64)
            self.ct_cnn._swish = nn.Identity()
        
        elif cnn == "efnb1":
            self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b1')
            self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 32, kernel_size = (3,3), stride = (2,2), 
                                                             bias = False, image_size = 512)
            self.ct_cnn._fc = nn.Linear(self.out_dict[cnn], 64)
            self.ct_cnn._swish = nn.Identity()

        elif cnn == "efnb2":
            self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b2')
            self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 32, kernel_size = (3,3), stride = (2,2), 
                                                             bias = False, image_size = 512)
            self.ct_cnn._fc = nn.Linear(self.out_dict[cnn], 64)
            self.ct_cnn._swish = nn.Identity()
        
        elif cnn == "efnb3":
            self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b3')
            self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 40, kernel_size = (3,3), stride = (2,2), 
                                                             bias = False, image_size = 512)
            self.ct_cnn._fc = nn.Linear(self.out_dict[cnn], 64)
            self.ct_cnn._swish = nn.Identity()
        
        elif cnn == "efnb4":
            self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b4')
            self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 48, kernel_size = (3,3), stride = (2,2), 
                                                             bias = False, image_size = 512)
            self.ct_cnn._fc = nn.Linear(self.out_dict[cnn], 64)
            self.ct_cnn._swish = nn.Identity()
        
        elif cnn == "efnb5":
            self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b5')
            self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 48, kernel_size = (3,3), stride = (2,2), 
                                                             bias = False, image_size = 512)
            self.ct_cnn._fc = nn.Linear(self.out_dict[cnn], 64)
            self.ct_cnn._swish = nn.Identity()
        
        elif cnn == "efnb6":
            self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b6')
            self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 56, kernel_size = (3,3), stride = (2,2), 
                                                             bias = False, image_size = 512)
            self.ct_cnn._fc = nn.Linear(self.out_dict[cnn], 64)
            self.ct_cnn._swish = nn.Identity()
        
        elif cnn == "efnb7":
            self.ct_cnn = EfficientNet.from_pretrained('efficientnet-b7')
            self.ct_cnn._conv_stem = Conv2dStaticSamePadding(1, 64, kernel_size = (3,3), stride = (2,2), 
                                                             bias = False, image_size = 512)
            self.ct_cnn._fc = nn.Linear(self.out_dict[cnn], 64)
            self.ct_cnn._swish = nn.Identity()
        
        else:
            raise ValueError("cnn not recognized")
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(64 + self.n_tab, 100)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(100, 3)
        self.relu2 = nn.ReLU() # quantile adjustment
        
        
    def forward(self, x_ct, x_tab, pid = None):
        
        if self.training:
            ct_f = self.ct_cnn(x_ct) # ct features
        else:
            # hashtable trick for inference
            # print('inference mode')
            if pid in self.cnn_features:
                ct_f = self.cnn_features[pid]
            else:
                ct_f = self.ct_cnn(x_ct)
                self.cnn_features[pid] = ct_f
                

        # print(ct_f.shape)
        # print(x_tab.shape)
        
        # concatenate
        x = torch.cat((ct_f, x_tab), -1) # concat on last axis
        
        # dropout
        if self.training:
             x = self.dropout(x)
                
        x = self.fc1(x)
        x = self.relu1(x)
        
        x1 = self.fc2(x)
        x2 = self.relu2(x1)
        
        x_f = x1 + torch.cumsum(x2, dim=-1)
        
        return x_f

from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

# tr_p, val_p = train_test_split(P, 
#                               shuffle=True, 
#                               train_size= 0.75) # 75% training




# dummy = False

# if dummy:
#     tr_p = tr_p[:4]
#     val_p = val_p[:4]

# osic_train = OSICData(tr_p, A, TAB)
# train_loader = DataLoader(osic_train, batch_size=4, shuffle=True, num_workers=4)

# osic_val = OSICData(val_p, A, TAB)
# val_loader = DataLoader(osic_val, batch_size=4, shuffle=True, num_workers=4)

# all data
# osic_all = OSICData(tr_p + val_p, A, TAB)
# all_loader = DataLoader(osic_all, batch_size=4, shuffle=True, num_workers=4) 

# score calculation

def score_np(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2)
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return np.mean(metric)


def score_avg(p, a): # patient id, predicted a
    percent_true = train.Percent.values[train.Patient == p]
    fvc_true = train.FVC.values[train.Patient == p]
    weeks_true = train.Weeks.values[train.Patient == p]

    fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]
    percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])
    return score_np(fvc_true, fvc, percent)

def rmse_avg(p, a): # patient id, predicted a
    percent_true = train.Percent.values[train.Patient == p]
    fvc_true = train.FVC.values[train.Patient == p]
    weeks_true = train.Weeks.values[train.Patient == p]

    fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]
    return mean_squared_error(fvc_true, fvc, squared = False)


# hyperparams

result_dir = hyp.results_dir

# training only resnet models on gpu 0
train_models = hyp.train_models 

# 'resnext101' -> seems too heavy for 1080
# 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'

# device
gpu = torch.device(f"cuda:{hyp.gpu_index}" if torch.cuda.is_available() else "cpu")


nfold = hyp.nfold # hyper


# loss function

def score(outputs,target):
    confidence = outputs[:,2] - outputs[:,0] # output -> 0-> 20% percentile, 50% percentile, 80% percentile
    clip = torch.clamp(confidence,min=70)
    target=torch.reshape(target,outputs[:,1].shape)
    delta = torch.abs(outputs[:,1] - target)
    delta = torch.clamp(delta,max=1000)
    sqrt_2 = torch.sqrt(torch.tensor([2.])).to(gpu)
    metrics = (delta*sqrt_2/clip) + torch.log(clip*sqrt_2)
    return torch.mean(metrics)

def qloss(outputs,target):
    qs = [0.2,0.5,0.8]
    qs = torch.tensor(qs,dtype=torch.float).to(gpu)
    e =  outputs - target
    e = e.to(gpu)
    v = torch.max(qs*e,(qs-1)*e)
    return torch.mean(v)


def hyb_loss(outputs,target,l):
    return l * qloss(outputs,target) + (1- l) * score(outputs,target)

# need to edit from here

for model in train_models:
    log = open(f"{result_dir}/{model}.txt", "a+")
    kfold =KFold(n_splits=nfold)
    
    ifold = 0
    for train_index, test_index in kfold.split(P):  
        # print(train_index, test_index) 

        p_train = np.array(P)[train_index] 
        p_test = np.array(P)[test_index] 
        
        osic_train = OSICData(p_train, A, TAB)
        train_loader = DataLoader(osic_train, batch_size=hyp.batch_size, shuffle=True, num_workers=4)

        osic_val = OSICData(p_test, A, TAB)
        val_loader = DataLoader(osic_val, batch_size=hyp.batch_size, shuffle=True, num_workers=4)
        
    
        tabct = TabCT(cnn = model).to(gpu)
        print(f"creating {model}")
        print(f"fold: {ifold}")
        log.write(f"fold: {ifold}\n")

        ifold += 1 


        n_epochs = hyp.n_epochs # max 30 epochs, patience 5, find the suitable epoch number for later final training

        best_epoch = n_epochs # 30


        optimizer = torch.optim.AdamW(tabct.parameters())
        criterion = torch.nn.L1Loss()

        max_score = 99999999.0000 # here, max score ]= minimum score

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in tqdm(enumerate(train_loader, 0)):

                [x, t], a, _ = data

                x = x.to(gpu)
                t = t.to(gpu)
                a = a.to(gpu)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = tabct(x, t)
                loss = criterion(outputs, a)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            print(f"epoch {epoch+1} train: {running_loss}")
            log.write(f"epoch {epoch+1} train: {running_loss}\n")

            running_loss = 0.0
            pred_a = {}
            for i, data in tqdm(enumerate(val_loader, 0)):

                [x, t], a, pid = data

                x = x.to(gpu)
                t = t.to(gpu)
                a = a.to(gpu)

                # forward
                outputs = tabct(x, t)
                loss = criterion(outputs, a)

                pids = pid
                preds_a = outputs.detach().cpu().numpy().flatten()

                for j, p_d in enumerate(pids):
                    pred_a[p_d] = preds_a[j]




                # print statistics
                running_loss += loss.item()
            print(f"epoch {epoch+1} val: {running_loss}")
            log.write(f"epoch {epoch+1} val: {running_loss}\n")
            # score calculation
            print(pred_a)
            print(len(pred_a))
            print(p_test)
            print(len(p_test))
            score_v = 0.
            rmse = 0.
            for p in p_test:
                score_v += (score_avg(p, pred_a[p]))/len(p_test)
                rmse += (rmse_avg(p, pred_a[p]))/len(p_test)

            print(f"val score: {score_v}")
            log.write(f"val score: {score_v}\n")
            log.write(f"val rmse: {rmse}\n")

            if score_v <= max_score:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': tabct.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'score': score_v
                    }, f"{result_dir}/{model}.tar")
                max_score = score_v
                best_epoch = epoch + 1
        # destroy model
        del tabct
        torch.cuda.empty_cache()

    # final training with optimized setting
    
    osic_all = OSICData(P, A, TAB)
    all_loader = DataLoader(osic_all, batch_size=2, shuffle=True, num_workers=4)

    # load the best model
    tabct = TabCT(cnn = model).to(gpu)
    tabct.load_state_dict(torch.load(f"{result_dir}/{model}.tar")["model_state_dict"])

    optimizer = torch.optim.AdamW(tabct.parameters(), lr = hyp.final_lr) # very small learning rate
    criterion = torch.nn.L1Loss()

    print(f"Final training")
    log.write(f"Final training\n")  
    for epoch in range(best_epoch + 2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in tqdm(enumerate(all_loader, 0)):

            [x, t], a, _ = data

            x = x.to(gpu)
            t = t.to(gpu)
            a = a.to(gpu)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = tabct(x, t)
            loss = criterion(outputs, a)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f"epoch {epoch+1} train: {running_loss}")
        log.write(f"epoch {epoch+1} train: {running_loss}\n") 

        torch.save({
            'epoch': best_epoch,
            'model_state_dict': tabct.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, f"{result_dir}/{model}.tar")

    print('Finished Training')
    # destroy model
    del tabct
    torch.cuda.empty_cache()




# ref: https://www.kaggle.com/miklgr500/linear-decay-based-on-resnet-cnn
# https://pytorch.org/docs/stable/index.html
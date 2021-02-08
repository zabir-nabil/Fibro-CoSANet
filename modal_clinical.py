
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


from torch import nn

# only clinical
class TabCT(nn.Module):
    def __init__(self, cnn, attn_filters, fc_dim, n_attn_layers):
        super(TabCT, self).__init__()

        
        self.fc_inter = nn.Linear(self.n_tab, self.fc_dim)

        self.fc = nn.Linear(self.fc_dim, 1)
        
    def forward(self, x_ct, x_tab):
                
        x = self.fc_inter(x_tab)

        x = self.fc(x)
        
        return x

from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error


# score calculation

def score(fvc_true, fvc_pred, sigma):
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
    return score(fvc_true, fvc, percent)

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

# removing noisy data
P = [p for p in P if p not in ['ID00011637202177653955184', 'ID00052637202186188008618']]

for model in train_models:
    for fd in hyp.fc_dim:
        for af in hyp.attn_filters:
            for nal in hyp.n_attn_layers:
                log = open(f"{result_dir}/{model}_fd_{fd}_af_{af}_nal_{nal}.txt", "a+")
                kfold =KFold(n_splits=nfold)
                
                ifold = 0
                for train_index, test_index in kfold.split(P):  
                    # print(train_index, test_index) 

                    p_train = np.array(P)[train_index] 
                    p_test = np.array(P)[test_index] 
                    
                    osic_train = OSICData(p_train, A, TAB)
                    train_loader = DataLoader(osic_train, batch_size=hyp.batch_size, shuffle=True, num_workers=hyp.num_workers)

                    osic_val = OSICData(p_test, A, TAB)
                    val_loader = DataLoader(osic_val, batch_size=hyp.batch_size, shuffle=True, num_workers=hyp.num_workers)
                    
                
                    tabct = TabCT(cnn = model, fc_dim = fd, attn_filters = af, n_attn_layers = nal).to(gpu)
                    print(f"creating {model} with {fd} feature_dim, {af} attn_filters, and {nal} n_attn_layers")
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
                        tabct.train()
                        for i, data in tqdm(enumerate(train_loader, 0)):

                            [x, t], a, _ = data
                            print(x.shape)
                            print(t.shape)

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
                        tabct.eval()
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
                                }, f"{result_dir}/{model}_fd_{fd}_af_{af}_nal_{nal}.tar")
                            max_score = score_v
                            best_epoch = epoch + 1
                    # destroy model
                    del tabct
                    torch.cuda.empty_cache()

                # final training with optimized setting
                
                osic_all = OSICData(P, A, TAB)
                all_loader = DataLoader(osic_all, batch_size=8, shuffle=True, num_workers=hyp.num_workers)

                # load the best model
                tabct = TabCT(cnn = model, fc_dim = fd, attn_filters = af, n_attn_layers = nal).to(gpu)
                tabct.load_state_dict(torch.load(f"{result_dir}/{model}_fd_{fd}_af_{af}_nal_{nal}.tar")["model_state_dict"])

                optimizer = torch.optim.AdamW(tabct.parameters(), lr = hyp.final_lr) # very small learning rate
                criterion = torch.nn.L1Loss()

                print(f"Final training")
                log.write(f"Final training\n")  
                for epoch in range(best_epoch + 2):  # loop over the dataset multiple times

                    running_loss = 0.0
                    tabct.train()
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
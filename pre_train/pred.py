import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torchmetrics
from model import MyNet
from load_data import CSSDatasetRetained1,CSSDatasetRetained2
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import random
import os
from torch.optim.swa_utils import AveragedModel, SWALR
from typing import Iterable
from math import cos, pi
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        from torch import optim
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=lr,weight_decay=0.001)#momentum=0.99
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr,amsgrad=True,weight_decay=1e-2)#1e-4
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr)#

        
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr,weight_decay=1e-2,momentum=0.99)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=3,
        #                                            verbose=False, threshold=1.5, threshold_mode='abs',
        #                                            cooldown=0, min_lr=1e-08, eps=1e-08)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 150)#下面也要该一改scheduler.step，epoch改为150
        

        self.device = device
        # print("它被调用了")

    def train(self, data_loader,epoch):
        # criterion = torch.nn.L1Loss()
        criterion=torch.nn.SmoothL1Loss()
       
        total_loss=0
        for i, data in enumerate(tqdm(data_loader)):
            data.to(self.device)
            y_hat1 = self.model(data)
            loss = criterion(y_hat1, data.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss=total_loss+loss
        # self.scheduler.step(total_loss/len(data_loader))
        self.scheduler.step()

        print(total_loss/len(data_loader))
        return 0

class Tester(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
   
    def test_regressor(self, data_loader):
        y_true = []
        y_pred = []
        smile_list=[]
        inchi_key_list=[]
        with torch.no_grad():
            for data in data_loader:
                data.to(self.device, non_blocking=True)
                y_hat = self.model(data)
                y_true.append(data.y)
                y_pred.append(y_hat)
                smile_list.append(data.smile)
                inchi_key_list.append(data.inchi_key)
            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)
            smile_list = np.concatenate(smile_list).tolist()
            inchi_key_list = np.concatenate(inchi_key_list).tolist()
            
            
            data = {
            "inchi_key":inchi_key_list ,
            "smile":smile_list,
            "From_RT": y_true.cpu().numpy(),
            "y_pred": y_pred.cpu().numpy(),
            }
            succ_data = pd.DataFrame(data)
            succ_data.to_csv('./pred_meto_test.csv')

        return 0;


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    num_works = 2
    test_batch = 8
    dataset_test = CSSDatasetRetained2('./SMRT_test')

    test_len = dataset_test.__len__()


    test_loader = DataLoader(dataset_test, batch_size=test_batch, shuffle=False,
                             num_workers=num_works, pin_memory=True, 
                             prefetch_factor=8, persistent_workers=True)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print('Creating a model.')

    model = MyNet()
    model.load_state_dict(torch.load('code/RT/pre_train/model/best_model.pth', map_location='cuda:1'))
    model.eval()
    tester = Tester(model, device)

    model.to(device=device)                                                                                                         
                
    tester.test_regressor(test_loader)






import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torchmetrics
from model import MyNet
from load_data_transfer import Transfer_data_load
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import random
import os
from argparse import ArgumentParser
from typing import Iterable
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        from torch import optim
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr,amsgrad=True,weight_decay=1e-2)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 150)
        self.device = device

    def train(self, data_loader,epoch):
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

        with torch.no_grad():
            for data in data_loader:
                data.to(self.device, non_blocking=True)
                y_hat = self.model(data)
                y_true.append(data.y)
                if len(y_hat.shape)==0:
                    y_hat=y_hat.view(-1)
                y_pred.append(y_hat)

            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

            mae=torch.abs(y_true - y_pred).mean()
            mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
            medAE = torch.median(torch.abs(y_true - y_pred))
            medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))
            
            score = torchmetrics.R2Score().to(self.device)
            r2 = score(y_pred, y_true)
        return mae,mre,medAE,medRE,r2
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def set_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
           

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:           
            continue
        for param in child.parameters():
            param.requires_grad = not freeze



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--DataSet', '-d', type=str,default="Eawag_XBridgeC18_364.xlsx")
    args = parser.parse_args()

    name = args.DataSet
    name1=name.split(".")[0]
    name2=name.split(".")[1]

    batch_size = 8
    num_works = 8
    lr = 0.0001
    epochs = 150
    test_batch = 8
    kfold = 10

    randints=[1,12,123,1234,12345]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_str="./transfer_data/reload/"+name1
    file_str="./results/"+name1+"_FE_"

    if True:
        print('Loading ...')
        dataset = Transfer_data_load('./transfer_data/reload/'+name1,name=name)
        print('Number of graphs in dataset: ', len(dataset))
        f_str=file_str

        for randint in randints:
            set_seed(randint)
            with open(f_str+str(randint)+'.txt', 'a') as f:
                result_head=[]
                result_last=[]
                for fold in range(kfold):
                    new_lr = lr
                    fold_size = len(dataset) // kfold
                    fold_reminder = len(dataset) % kfold
                    split_list = [fold_size] * kfold            
                    for reminder in range(fold_reminder):
                        split_list[reminder] = split_list[reminder] + 1
                    split = random_split(dataset, split_list)
                    best_test_mae = 9999999


                    model = MyNet()
                    model.load_state_dict(torch.load('./model_dict/best_model.pth', map_location='cuda:0'))
                    set_freeze(model=model)
                    set_freeze_by_names(model, ['in_node','in_edge','conv1','conv2','conv3','conv4','conv5','conv6'])


                    model.to(device=device)
                    trainer = Trainer(model, new_lr, device)
                    tester = Tester(model, device)

                    torch.cuda.empty_cache()
                    test_dataset = split[fold]

                    train_list = []
                    for m in range(kfold):
                        if m != fold:
                            train_list.append(split[m])
                    train_dataset = torch.utils.data.ConcatDataset(train_list)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=num_works, pin_memory=True,
                                                prefetch_factor=4)
                    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True,
                                                num_workers=num_works, pin_memory=True,
                                                prefetch_factor=4)
                    f.write(f'randint{randint}\n')
                    for epoch in range(epochs):
                        model.eval()
                        trainer.train(train_loader,epoch)
                        print(trainer.optimizer.param_groups[0]['lr'])
                        model.eval()
                        mae_train,mre_train,medAE_train,medRE_train,r2_train = tester.test_regressor(train_loader)
                        mae_dev,mre_dev,medAE_dev,medRE_dev,r2_dev = tester.test_regressor(test_loader)
                        if mae_dev < best_test_mae:
                            best_test_mae = mae_dev
                        print(f'kfold:{fold}\tepoch:{epoch}\ttrain_loss:{mae_train}\tmre_train:{mre_train}\tmedAE_train:{medAE_train}\tmedRE_train:{medRE_train}\tr2_train:{r2_train}')
                        print(f'kfold:{fold}\tepoch:{epoch}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}')
                        f.write(f'kfold:{fold}\tepoch:{epoch}\ttrain_loss:{mae_train}\tmre_train:{mre_train}\tmedAE_train:{medAE_train}\tmedRE_train:{medRE_train}\tr2_train:{r2_train}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}\tbest:{best_test_mae}\n')
                        f.flush()
                    f.write(f'randint{randint}\tbest:{best_test_mae}\n')

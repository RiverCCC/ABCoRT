import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torchmetrics
from model import MyNet
from load_data_SMRT import SMRT_Dataset_Load_train
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import random
import os
from torch import optim
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        
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

if __name__ == '__main__':
    batch_size = 64
    num_works = 2
    lr = 1e-5
    epochs = 150
    test_batch = 64

    randint=1

    set_seed(randint)
    print("loading... get SMRT train data feature")
    dataset_train = SMRT_Dataset_Load_train('./SMRT_data/reload/SMRT_train')

    train_len = dataset_train.__len__()
    train_len2 = int(dataset_train.__len__() * 0.9)

    dev_len = train_len - train_len2
    train_dataset, dev_dataset = random_split(dataset_train, [train_len2, dev_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_works, pin_memory=True, 
                              prefetch_factor=2, persistent_workers=True)
    dev_loader = DataLoader(dev_dataset, batch_size=test_batch, shuffle=True,
                            num_workers=num_works, pin_memory=True, 
                            prefetch_factor=8, persistent_workers=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    print(f'use\r', device)
    print('-' * 100)
    print('# of training data samples:', len(train_dataset))
    print('# of deving data samples:', len(dev_dataset))
    print('-' * 100)
    print('Creating a model.')

    model = MyNet()
    trainer = Trainer(model, lr, device)
    tester = Tester(model, device)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)
    print('Start training.')


    model.to(device=device)                                                                                                         

    mae_test_best = 40
    with open('./results/SMRT_result.txt', 'a') as f:
        for epoch in range(epochs):
            model.train()
            try:
                loss_training = trainer.train(train_loader,epoch)
                print(trainer.optimizer.param_groups[0]['lr'])
                model.eval()
                mae_train,mre_train,medAE_train,medRE_train,r2_train = tester.test_regressor(train_loader)
                mae_dev ,mre_dev,medAE_dev,medRE_dev,r2_dev = tester.test_regressor(dev_loader)
                print(f'epoch:{epoch}\ttrain_loss:{mae_train}\tmre_train:{mre_train}\tmedAE_train:{medAE_train}\tmedRE_train:{medRE_train}\tr2_train:{r2_train}')
                print(f'epoch:{epoch}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}')
                f.write(f'epoch:{epoch}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}\n')
                f.flush()

                if mae_dev < mae_test_best:
                    torch.save(model.state_dict(),f'./model_dict/best_model_SMRT.pkl')
                    mae_test_best = mae_dev


            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise Exception
        # torch.save(model.state_dict(),f'./model/last_model_SMRT.pkl')


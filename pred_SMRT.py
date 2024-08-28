from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torchmetrics
from model import MyNet
from load_data_SMRT import SMRT_Dataset_Load_test
import torch
from torch_geometric.loader import DataLoader
import warnings
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
                y_pred.append(y_hat)
            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

            mae=torch.abs(y_true - y_pred).mean()
            mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
            medAE = torch.median(torch.abs(y_true - y_pred))
            medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))
            score = torchmetrics.R2Score().to(self.device)
            r2 = score(y_pred, y_true)            
            
        return mae,mre,medAE,medRE,r2;



if __name__ == '__main__':

    num_works = 1
    test_batch = 64
    dataset_test = SMRT_Dataset_Load_test('./SMRT_data/reload/SMRT_test')

    test_len = dataset_test.__len__()


    test_loader = DataLoader(dataset_test, batch_size=test_batch, shuffle=False,
                             num_workers=num_works, pin_memory=True, 
                             prefetch_factor=8, persistent_workers=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Creating a model.')

    model = MyNet()
    model.load_state_dict(torch.load('./model_dict/best_model.pth', map_location='cuda:0'))
    model.eval()
    tester = Tester(model, device)

    model.to(device=device)                                                                                                         
                

    mae_test,mre_test,medAE_test,medRE_test,r2_test=tester.test_regressor(test_loader)
    print(f'test_loss:{mae_test}\tmre_dev:{mre_test}\tmedAE_dev:{medAE_test}\tmedRE_dev:{medRE_test}\tr2_dev:{r2_test}')






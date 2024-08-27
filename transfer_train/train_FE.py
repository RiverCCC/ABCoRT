import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torchmetrics
from model import MyNet
from load_data import CSSDatasetRetained1
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
        # criterion=torch.nn.L1Loss(reduction="mean")
        # loss=0
        with torch.no_grad():
            for data in data_loader:
                data.to(self.device, non_blocking=True)
                y_hat = self.model(data)
                # loss += criterion(y_hat, data.y).item()

                # total_loss += torch.abs(y_hat - data.y).sum()
                # mre_total = torch.div(torch.abs(y_hat - data.y), data.y).sum()

                y_true.append(data.y)
                if len(y_hat.shape)==0:
                    y_hat=y_hat.view(-1)
                y_pred.append(y_hat)

            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

            # mae = torch.abs(y_true - y_pred).sum()
            mae=torch.abs(y_true - y_pred).mean()
            # mae=loss/len(data_loader)
            mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
            medAE = torch.median(torch.abs(y_true - y_pred))
            medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))
            
            score = torchmetrics.R2Score().to(self.device)
            r2 = score(y_pred, y_true)
        # return mae.item(), medAE.item(), mre.item(), medRE.item(), r2.item()
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



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    batch_size = 8
    num_works = 8
    lr = 0.0001
    # lr=0.001
    epochs = 150#.......................................................................
    test_batch = 8
    kfold = 10

    
    # randints=[1,12,123,1234,12345]
    randints=[1,12,123]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_str=[
                # 'code/RT/transfer_data/Eawag_XBridgeC','code/RT/transfer_data/FEM_lipids_72','code/RT/transfer_data/FEM_long_412',
                # 'code/RT/transfer_data/IPB_Halle_82','code/RT/transfer_data/LIFE_new_184','code/RT/transfer_data/LIFE_old_194',
                'code/RT/transfer_data/UniToyama_Atlantis_143']
    file_str=[
            # 'code/RT/transfer_train/results_real/try_EXC_FE_','code/RT/transfer_train/results_real/try_FEM72_FE_',
            #     'code/RT/transfer_train/results_real/try_FEM412_FE_','code/RT/transfer_train/results_real/try_IPB82_FE_',
            #     'code/RT/transfer_train/results_real/try_LIFE184_FE_','code/RT/transfer_train/results_real/try_LIFE194_FE_',
                'code/RT/transfer_train/results_real/try_UA143_FE_']

    # dataset = CSSDatasetRetained1('code/RT/transfer_data/Eawag_XBridgeC')
    # dataset = CSSDatasetRetained1('code/RT/transfer_data/FEM_lipids_72')
    # dataset = CSSDatasetRetained1('code/RT/transfer_data/FEM_long_412')
    # dataset = CSSDatasetRetained1('code/RT/transfer_data/IPB_Halle_82')
    # dataset = CSSDatasetRetained1('code/RT/transfer_data/LIFE_new_184')
    # dataset = CSSDatasetRetained1('code/RT/transfer_data/LIFE_old_194')
    # dataset = CSSDatasetRetained1('code/RT/transfer_data/UniToyama_Atlantis_143')


    # with open('code/RT/transfer_train/results_real/try_EXC_FE.txt', 'a') as f:
    # with open('code/RT/transfer_train/results_real/try_FEM72_FE.txt', 'a') as f:
    # with open('code/RT/transfer_train/results_real/try_FEM412_FE.txt', 'a') as f:
    # with open('code/RT/transfer_train/results_real/try_IPB82_FE.txt', 'a') as f:
    # with open('code/RT/transfer_train/results_real/try_LIFE184_FE.txt', 'a') as f:
    # with open('code/RT/transfer_train/results_real/try_LIFE194_FE.txt', 'a') as f:
    # with open('code/RT/transfer_train/results_real/try_UA143_FE.txt', 'a') as f:
    for num in range(len(dataset_str)):
        dataset = CSSDatasetRetained1(dataset_str[num])
        f_str=file_str[num]
        for randint in randints:
            set_seed(randint)
            with open(f_str+str(randint)+'.txt', 'a') as f:
                result_head=[]
                result_last=[]
                for fold in range(kfold):
                    # fold=2
                    new_lr = lr

                    fold_size = len(dataset) // kfold
                    fold_reminder = len(dataset) % kfold
                    split_list = [fold_size] * kfold
                    
                    for reminder in range(fold_reminder):
                        split_list[reminder] = split_list[reminder] + 1


                    split = random_split(dataset, split_list)
                    # print(split_list)
                    # print(1/0)
                    best_test_mae = 9999999


                    #特征提取FE
                    model = MyNet()
                    model.load_state_dict(torch.load('code/RT/transfer_train/best_model.pth', map_location='cuda:0'))
                    set_freeze(model=model)
                    set_freeze_by_names(model, ['in_node','in_edge','conv1','conv2','conv3','conv4','conv5','conv6'])


                    model.to(device=device)
                    trainer = Trainer(model, new_lr, device)
                    tester = Tester(model, device)

                    torch.cuda.empty_cache()
                    test_dataset = split[fold]
                    # print(test_dataset[0])

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
                        
                        # model.train()#..................................................................................

                        model.eval()#。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
                        trainer.train(train_loader,epoch)
                        print(trainer.optimizer.param_groups[0]['lr'])
                        model.eval()
                        mae_train,mre_train,medAE_train,medRE_train,r2_train = tester.test_regressor(train_loader)
                        mae_dev,mre_dev,medAE_dev,medRE_dev,r2_dev = tester.test_regressor(test_loader)
                        if mae_dev < best_test_mae:
                            best_test_mae = mae_dev
                        #     torch.save(model, f'code/RT/transfer_train/model/EXC_FE_A_fold{fold}.pth')
                            # torch.save(model, f'code/RT/transfer_train/model/FEM72_FE_fold{fold}.pth')
                            # torch.save(model, f'code/RT/transfer_train/model/FEM412_FE_fold{fold}.pth')
                            # torch.save(model, f'code/RT/transfer_train/model/IPB82_FE_fold{fold}.pth')
                            # torch.save(model, f'code/RT/transfer_train/model/LIFE184_FE_fold{fold}.pth')
                            # torch.save(model, f'code/RT/transfer_train/model/LIFE194_FE_fold{fold}.pth')
                            # torch.save(model, f'code/RT/transfer_train/model/UA143_FE_fold{fold}.pth')



                        print(f'kfold:{fold}\tepoch:{epoch}\ttrain_loss:{mae_train}\tmre_train:{mre_train}\tmedAE_train:{medAE_train}\tmedRE_train:{medRE_train}\tr2_train:{r2_train}')
                        print(f'kfold:{fold}\tepoch:{epoch}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}')
                        f.write(f'kfold:{fold}\tepoch:{epoch}\ttrain_loss:{mae_train}\tmre_train:{mre_train}\tmedAE_train:{medAE_train}\tmedRE_train:{medRE_train}\tr2_train:{r2_train}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}\tbest:{best_test_mae}\n')
                        f.flush()
                    f.write(f'randint{randint}\tbest:{best_test_mae}\n')
                    result_head.append(best_test_mae)
                    result_last.append(mae_dev)
                f.write(f'results:{result_head}\n')
                f.write(f'result_last:{result_last}\n')
                avg=sum(result_head)/10
                avg_last=sum(result_last)/10
                f.write(f'avg:{avg}\tavg_last:{avg_last}\n')
                
            

        


















            # if mae_dev < best_test_mae:
            #     best_test_mae = mae_dev
            #     best_test_medAE = medAE_test
            #     best_test_mre = mre_test
            #     best_test_medRE = medRE_test
            #     r2_test_best = r2_test
            #     torch.save(model, f'./models/{dataset_name}/fold{fold}.pkl')

        # print(f'time:{time.time()-time_start:.2f},dataset:{dataset_name},fold:{fold},mae:{best_test_mae:.2f},mre:{best_test_mre * 100:.2f},medAE:{best_test_medAE:.2f},medRE:{best_test_medRE * 100:.2f},r2:{r2_test_best:.2f}')
        # file.write(f'{seed},{dataset_name},{fold},{best_test_mae},{best_test_mre * 100},{best_test_medAE:},{best_test_medRE * 100},{r2_test_best}\n')
        # file.flush()







    # train_len = dataset_train.__len__()
    # train_len2 = int(dataset_train.__len__() * 0.9)


    # dev_len = train_len - train_len2
    # train_dataset, dev_dataset = random_split(dataset_train, [train_len2, dev_len])

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                           num_workers=num_works, pin_memory=True, 
    #                           prefetch_factor=8, persistent_workers=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=test_batch, shuffle=True,
    #                         num_workers=num_works, pin_memory=True, 
    #                         prefetch_factor=8, persistent_workers=True)
    
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # torch.cuda.max_memory_allocated(device) 
    # torch.cuda.max_memory_cached(device)

    # print(f'use\r', device)
    # print('-' * 100)
    # print('The preprocess has finished!')
    # print('# of training data samples:', len(train_dataset))
    # print('# of deving data samples:', len(dev_dataset))
    # # print('# of test data samples:', len(dataset_test))
    # print('-' * 100)
    # print('Creating a model.')

    # model = MyNet()
    # # model = AveragedModel(model)
    # # model = torch.load('./model/best_model2.pkl', map_location='cuda:0')
    # # model = torch.load('best_model_epoch102_loss_32.12735703089935.pkl',map_location='cuda:0')
    # # for m in model.modules():
    # #     if isinstance(m, torch.nn.Linear):
    # #         torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
    # # model = torch.load('code/me/uploadsss/rts5/model/best_model16_3.pkl', map_location='cuda:1')
    # trainer = Trainer(model, lr, device)
    # tester = Tester(model, device)
    # print('# of model parameters:',
    #       sum([np.prod(p.size()) for p in model.parameters()]))
    # print('-' * 100)
    # print('Start training.')
    # print('The result is saved in the output directory every epoch!')

    # np.random.seed(randint)
    # torch.manual_seed(randint)

    # model.to(device=device)                                                                                                         


    # mae_test_best = 27
    # with open('code/me/rts2/results/try1_noload.txt', 'a') as f:
    #     for epoch in range(epochs):
            
    #         model.train()
    #         try:
                
    #             loss_training = trainer.train(train_loader,epoch)
    #             print(trainer.optimizer.param_groups[0]['lr'])

    #             model.eval()
    #             # if epoch%5 == 0:
    #             if True:
    #                 loss_train,mre_train,medAE_train,medRE_train,r2_train = tester.test_regressor(train_loader)
    #                 loss_dev,mre_dev,medAE_dev,medRE_dev,r2_dev = tester.test_regressor(dev_loader)

    #                 mae_train= loss_train.item() / train_len2
    #                 mae_dev = loss_dev.item() / dev_len
    #                 # mae_test=loss_test.item()/ test_len 
    #                 print(f'epoch:{epoch}\ttrain_loss:{mae_train}\tmre_train:{mre_train}\tmedAE_train:{medAE_train}\tmedRE_train:{medRE_train}\tr2_train:{r2_train}')
    #                 print(f'epoch:{epoch}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}')
                    

    #                 f.write(f'epoch:{epoch}\ttrain_loss:{mae_train}\tmre_train:{mre_train}\tmedAE_train:{medAE_train}\tmedRE_train:{medRE_train}\tr2_train:{r2_train}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}\n')
    #                 f.flush()
    #             else:

    #                 loss_dev,mre_dev,medAE_dev,medRE_dev,r2_dev = tester.test_regressor(dev_loader)
    #                 loss_test,mre_test,medAE_test,medRE_test,r2_test=tester.test_regressor(test_loader)

    #                 mae_dev = loss_dev.item() / dev_len
    #                 mae_test=loss_test.item()/ test_len 
                    
    #                 print(f'epoch:{epoch}\tdev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}')
    #                 print(f'epoch:{epoch}\ttest_loss:{mae_test}\tmre_dev:{mre_test}\tmedAE_dev:{medAE_test}\tmedRE_dev:{medRE_test}\tr2_dev:{r2_test}')

    #                 f.write(f'dev_loss:{mae_dev}\tmre_dev:{mre_dev}\tmedAE_dev:{medAE_dev}\tmedRE_dev:{medRE_dev}\tr2_dev:{r2_dev}\ttest_loss:{mae_test}\tmre_dev:{mre_test}\tmedAE_dev:{medAE_test}\tmedRE_dev:{medRE_test}\tr2_dev:{r2_test}\n')
    #                 f.flush()

    #             if mae_dev < mae_test_best:
    #                 # torch.save(model, f'./model/best_ model_epoch{epoch}_loss_{mae_dev}.pkl')
    #                 torch.save(model, f'code/me/rts2/model/best_model1_noload.pkl')
    #                 mae_test_best = mae_dev


    #         except RuntimeError as exception:
    #             if "out of memory" in str(exception):
    #                 print("WARNING: out of memory")
    #                 if hasattr(torch.cuda, 'empty_cache'):
    #                     torch.cuda.empty_cache()
    #             else:
    #                 raise Exception

            

    # model = torch.load('code/me/rts2/model/best_model1_noload.pkl', map_location='cuda:1')
    # tester.test_regressor(dev_loader)



from typing import  Union, List, Tuple
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import pickle
from get_feature import get_AtomBond_feature
class SMRT_Dataset_Load_train(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'SMRT_train.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass


    def process(self):
        
        res = pd.read_csv("./SMRT_data/data/SMRT_train.csv")
        y = res['rt']
        smile_list = res['smile']
        data_list = []
        succ_inchi, succ_rt, success_index, atom_feature,edge_index,edge_attr = get_AtomBond_feature(smile_list, y,type=0)


        for index, smile in enumerate(smile_list):
            data = Data(x=atom_feature[index], y=torch.tensor(res['rt'][index], dtype=torch.float32), 
            edge_index=edge_index[index],edge_attr=edge_attr[index])
            data_list.append(data)


        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
       
        torch.save((data, slices), self.processed_paths[0])

class SMRT_Dataset_Load_test(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'SMRT_test.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass


    def process(self):
        res = pd.read_csv("./SMRT_data/data/SMRT_test.csv")
        y = res['rt']
        smile_list = res['smile']
        succ_inchi, succ_rt, success_index, atom_feature,edge_index,edge_attr = get_AtomBond_feature(smile_list,y,type=0)
        data_list = []
        for index, inchi in enumerate(smile_list):
            data = Data(x=atom_feature[index], y=torch.tensor(res['rt'][index], dtype=torch.float32), 
            edge_index=edge_index[index],edge_attr=edge_attr[index])  
            data_list.append(data)

        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list) 
        torch.save((data, slices), self.processed_paths[0])
if __name__ == '__main__':

    print('Loading ...get SMRT train feature')
    dataset1 =SMRT_Dataset_Load_train('./SMRT_data/reload/SMRT_train')
    print('Number of graphs in dataset: ', len(dataset1))
    print('Loading ...get SMRT test feature')
    dataset2 = SMRT_Dataset_Load_test('./SMRT_data/reload/SMRT_test')
    print('Number of graphs in dataset: ', len(dataset2))






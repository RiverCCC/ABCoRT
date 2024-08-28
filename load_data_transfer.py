
from typing import  Union, List, Tuple
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import pickle
import pandas as pd
from get_feature import get_AtomBond_feature
from argparse import ArgumentParser
class Transfer_data_load(InMemoryDataset):
    def __init__(self, root,name, transform=None, pre_transform=None, pre_filter=None):
        self.name=name
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
        return ''

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass


    def process(self):
        if self.name.endswith(".xlsx"):
            res = pd.read_excel("./transfer_data/data/"+self.name)
        elif self.name.endswith(".csv"):
            res = pd.read_csv("./transfer_data/data/"+self.name)

        
        if self.name == "MetaboBase.csv" or self.name=="RIKEN_MONA.xlsx":
            temp=0
            y = res['RT']
            inchi_list = res['SMILES']
        else:
            temp=1
            y = res['RT']
            inchi_list = res['InChI']

        data_list = []
        succ_inchi, succ_rt, success_index, atom_feature,edge_index,edge_attr = get_AtomBond_feature(inchi_list, y,temp)

        if self.name == "MetaboBase.csv" or self.name=="RIKEN_MONA.xlsx" or self.name=="MassBank1.csv" or self.name=="MassBank2.csv":
            for index, inchi in enumerate(inchi_list):
                data = Data(x=atom_feature[index], y=torch.tensor(res['RT'][index], dtype=torch.float32), edge_index=edge_index[index],edge_attr=edge_attr[index],
                inchi=inchi)
                data_list.append(data)
        else:
            for index, inchi in enumerate(inchi_list):
                data = Data(x=atom_feature[index], y=torch.tensor(res['RT'][index]*60, dtype=torch.float32), edge_index=edge_index[index],edge_attr=edge_attr[index],
                inchi=inchi)
                data_list.append(data)

        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--DataSet', '-d', type=str,default="Eawag_XBridgeC18_364.xlsx")
    
    args = parser.parse_args()

    name = args.DataSet
    name1=name.split(".")[0]
    name2=name.split(".")[1]
    print('Loading ...')
    dataset1 = Transfer_data_load('./transfer_data/reload/'+name1,name=name)
    print('Number of graphs in dataset: ', len(dataset1))






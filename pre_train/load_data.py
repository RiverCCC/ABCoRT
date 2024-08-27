#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Ipynb_importer
from typing import Optional, Callable, Union, List, Tuple
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data, in_memory_dataset, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np
import os


from torch_geometric.utils import to_networkx, to_dense_adj
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import random
import os.path as osp
import rdkit

from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.preprocessing import OneHotEncoder
from rdkit.Chem.Scaffolds import MurckoScaffold
import sys
import tqdm
# import feature_ops


# In[4]:


class CSSDatasetRetained1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])#加载到内存

    #@property形成一个不能随意修改的变量
    #在root目录下原始数据文件所在的文件夹
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    
    #在root目录下把原始数据文件处理后的文件所在文件夹
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'more_300_SMRT_data.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass


    def process(self):
        # res = pd.read_csv(os.path.join(self.raw_dir,self.raw_file_names), sep=';')
        res = pd.read_csv("/home/lwh/anaconda3/man/manzzz/data/SMRT_train.csv")
        #pubchem，rt,inchi
        y = res['rt']
        inchi_list = res['smile']
        # indexs=res["index"]
        data_list = []


        with open('anaconda3/man/manzzz/data_information/atom_feature13/all_atom_feature13_noH_train.txt', 'rb') as text:
            atom_feature = pickle.load(text)
        
        with open('anaconda3/man/manzzz/data_information/atom_feature13/all_edge_attr13_noH_train.txt', 'rb') as text:
            edge_attr = pickle.load(text)
        
        with open('anaconda3/man/manzzz/data_information/atom_feature13/all_edge_index13_noH_train.txt', 'rb') as text:
            edge_index = pickle.load(text)
        
  

        for index, inchi in enumerate(inchi_list):
            if True:
                data = Data(x=atom_feature[index], y=torch.tensor(res['rt'][index], dtype=torch.float32), 
                edge_index=edge_index[index],edge_attr=edge_attr[index])
                data_list.append(data)
                # print(data)
            # except:
            #     print(str(index)+"处理出错")
            #     pass

        print(data_list.__len__())#一个分子一个图，有很多个图
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        #作用就是通过self.collate把数据划分成不同slices去保存读取 （大数据块切成小块）
        #所以即使只有一个graph写成了data， 在调用self.collate时，也要写成list:
        
        #collate()函数接收一个列表的Data对象，
        #返回合并后的Data对象以及用于从合并后的Data对象重构各个原始Data对象的切片字典slices。
        #最后我们将这个巨大的Data对象和切片字典slices保存到文件。
        torch.save((data, slices), self.processed_paths[0])
        #processed_paths()属性方法是在基类（DataSet）中定义的，
        #它对self.processed_dir文件夹与processed_file_names()属性方法的返回每一个文件名做拼接，然后返回。
        #返回的是一个列表，所以要加[0]
        #这个函数就是返回经过处理后的数据文件的地址

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()

class CSSDatasetRetained2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])#加载到内存

    #@property形成一个不能随意修改的变量
    #在root目录下原始数据文件所在的文件夹
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    
    #在root目录下把原始数据文件处理后的文件所在文件夹
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'more_300_SMRT_data.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass


    def process(self):
        # res = pd.read_csv(os.path.join(self.raw_dir,self.raw_file_names), sep=';')
        res = pd.read_csv("/home/lwh/anaconda3/man/manzzz/data/SMRT_test.csv")
        #pubchem，rt,inchi
        y = res['rt']
        inchi_list = res['smile']
        # indexs=res["index"]
        data_list = []


        with open('anaconda3/man/manzzz/data_information/atom_feature13/all_atom_feature13_noH_test.txt', 'rb') as text:
            atom_feature = pickle.load(text)
        
        with open('anaconda3/man/manzzz/data_information/atom_feature13/all_edge_attr13_noH_test.txt', 'rb') as text:
            edge_attr = pickle.load(text)
        
        with open('anaconda3/man/manzzz/data_information/atom_feature13/all_edge_index13_noH_test.txt', 'rb') as text:
            edge_index = pickle.load(text)
       


        for index, inchi in enumerate(inchi_list):
            if True:
                data = Data(x=atom_feature[index], y=torch.tensor(res['rt'][index], dtype=torch.float32), 
                edge_index=edge_index[index],edge_attr=edge_attr[index])
                
                
                data_list.append(data)
                # print(data)
            # except:
            #     print(str(index)+"处理出错")
            #     pass

        print(data_list.__len__())#一个分子一个图，有很多个图
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        #作用就是通过self.collate把数据划分成不同slices去保存读取 （大数据块切成小块）
        #所以即使只有一个graph写成了data， 在调用self.collate时，也要写成list:
        
        #collate()函数接收一个列表的Data对象，
        #返回合并后的Data对象以及用于从合并后的Data对象重构各个原始Data对象的切片字典slices。
        #最后我们将这个巨大的Data对象和切片字典slices保存到文件。
        torch.save((data, slices), self.processed_paths[0])
        #processed_paths()属性方法是在基类（DataSet）中定义的，
        #它对self.processed_dir文件夹与processed_file_names()属性方法的返回每一个文件名做拼接，然后返回。
        #返回的是一个列表，所以要加[0]
        #这个函数就是返回经过处理后的数据文件的地址

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()

    


if __name__ == '__main__':

    print('Loading ...')
    dataset1 = CSSDatasetRetained1('/home/lwh/anaconda3/man/manzzz/SMRT13_train')
    dataset2 = CSSDatasetRetained2('/home/lwh/anaconda3/man/manzzz/SMRT13_test')
    print('Number of graphs in dataset: ', len(dataset1))
    print('Number of graphs in dataset: ', len(dataset2))

    # print(dataset.get(0).fingerprint)
    # dataset.plot_graph(len(dataset)-1)


# In[ ]:





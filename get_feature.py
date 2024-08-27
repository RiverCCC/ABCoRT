import numpy as np
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
import pickle
from tqdm import *
from rdkit import  Chem
from feature_ops import atom_featurizer,bond_featurizer
import torch

def get_AtomBond_feature(inchis, y,type):

    succ_inchis = []
    succ_index = []
    succ_rt = []
    atom_feature=[]
    edge_index_all=[]
    edge_attr_all=[]
    INDEX = -1

    for inchi in tqdm(inchis):
        INDEX += 1
        try:
            if type==0:
                mol = Chem.MolFromSmiles(inchi)
            elif type ==1:
                mol = Chem.MolFromInchi(inchi)
            else:
                print("please input type")
            node_features = np.array([atom_featurizer(atom) for atom in mol.GetAtoms()], dtype='float32')
            x1= torch.tensor(node_features, dtype=torch.float32)

            
            row, col, edge_feat = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_features = bond_featurizer(bond)
                edge_feat.append(bond_features)
                edge_feat.append(bond_features)

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
            
        except:
            print(str(INDEX)+" error")
            continue
        succ_inchis.append(inchi)
        succ_index.append(INDEX)
        succ_rt.append(y[INDEX])
        atom_feature.append(x1)
        edge_index_all.append(edge_index)
        edge_attr_all.append(edge_attr)

    return succ_inchis, succ_rt, succ_index, atom_feature,edge_index_all,edge_attr_all

# if __name__ == '__main__':
    
#     res = pd.read_excel('./data/UniToyama_Atlantis_143.xlsx')
#     y = res['RT']
#     inchi_list = res['InChI']
#     succ_inchi, succ_rt, success_index, atom_feature,edge_index_all,edge_attr_all = get_atom_feature(inchi_list, y)

#     text_txt = './data_information/UniToyama_Atlantis_143/atom_feature_noH.txt'
#     with open(text_txt, 'wb') as text:
#         pickle.dump(atom_feature, text)
        
#     text_txt2='./data_information/UniToyama_Atlantis_143/edge_index_noH.txt'
#     with open(text_txt2, 'wb') as text:
#         pickle.dump(edge_index_all, text)
        
#     text_txt3='./data_information/UniToyama_Atlantis_143/edge_attr_noH.txt'
#     with open(text_txt3, 'wb') as text:
#         pickle.dump(edge_attr_all, text)








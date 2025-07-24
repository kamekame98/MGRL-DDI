from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np
import csv
import dgl


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

class DrugDataset(Dataset):
    def __init__(self, data_df, drug_atom, drug_motif_noglobal):
        self.data_df = data_df
        self.drug_atom = drug_atom
        self.drug_motif_noglobal = drug_motif_noglobal


    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    def collate_fn(self, batch):
        head_list_atom = []
        head_list_motif_noglobal = []
        tail_list_atom = []
        tail_list_motif_noglobal = []
        motif_bipart = []
        label_list = []

        for row in batch:
            Drug1_smiels, Drug2_smiels, Y = row['drug_A'], row['drug_B'], row['DDI']
            head_list_atom.append(self.drug_atom.get(Drug1_smiels))
            head_list_motif_noglobal.append(self.drug_motif_noglobal.get(Drug1_smiels))
            tail_list_atom.append(self.drug_atom.get(Drug2_smiels))
            tail_list_motif_noglobal.append(self.drug_motif_noglobal.get(Drug2_smiels))
            mol_1_x = self.drug_motif_noglobal.get(Drug1_smiels).x
            mol_2_x = self.drug_motif_noglobal.get(Drug2_smiels).x
            motif_bipart.append(self._create_b_graph(get_bipartite_graph(mol_1_x, mol_2_x), mol_1_x, mol_2_x))
            label_list.append(torch.Tensor([Y]))

        head_atom = Batch.from_data_list(head_list_atom, follow_batch=['edge_index'])
        tail_atom = Batch.from_data_list(tail_list_atom, follow_batch=['edge_index'])

        head_motif_noglobal = Batch.from_data_list(head_list_motif_noglobal)
        tail_motif_noglobal = Batch.from_data_list(tail_list_motif_noglobal)

        motif_bipart = Batch.from_data_list(motif_bipart)
        label = torch.cat(label_list, dim=0)

        return head_atom, tail_atom, head_motif_noglobal, tail_motif_noglobal, motif_bipart, label

    def _create_b_graph(self, edge_index, x_s, x_t):
        return BipartiteData(edge_index, x_s, x_t)


def get_bipartite_graph(mol_node_1, mol_node_2):
    num_nodes_1 = mol_node_1.size(0)  
    num_nodes_2 = mol_node_2.size(0)
    x1 = np.arange(0, num_nodes_1)
    x2 = np.arange(0, num_nodes_2)
    edge_list = torch.LongTensor(np.meshgrid(x1, x2))
    edge_list = torch.stack([edge_list[0].reshape(-1), edge_list[1].reshape(-1)])

    return edge_list


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def load_ddi_dataset(root, batch_size):
    drug_atom = read_pickle(os.path.join(root, 'drug_data_pyg_atom.pkl'))
    drug_motif_noglobal = read_pickle(os.path.join(root, 'drug_data_pyg_motif.pkl'))

    train_df = pd.read_csv(os.path.join(root, f'train.csv'))
    val_df = pd.read_csv(os.path.join(root, f'val.csv'))
    test_df = pd.read_csv(os.path.join(root, f'test.csv'))

    train_set = DrugDataset(train_df, drug_atom, drug_motif_noglobal)
    val_set = DrugDataset(val_df, drug_atom, drug_motif_noglobal)
    test_set = DrugDataset(test_df, drug_atom, drug_motif_noglobal)

    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the val set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader



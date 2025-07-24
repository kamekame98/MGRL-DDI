from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from torch import nn
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import ast
from rdkit import DataStructs
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_scatter import scatter_add
from tqdm import tqdm
from operator import index
import torch
import pickle
from collections import Counter
import torch.utils.data
import os
import torch.nn.functional as F
import dgl
from scipy import sparse as sp
import numpy as np
from utils import Mol_Tokenizer


class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)


def one_of_k_encoding_atom(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]

def one_of_k_encoding_unk_atom(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s,
                    allowable_set))

def atom_features_atom(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk_atom(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding_atom(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk_atom(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk_atom(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + one_of_k_encoding_unk_atom(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk_atom(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)

def edge_features_atom(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()

def adjust_tensor_size(tensor, target_shape, pad_value):
    current_shape = tensor.shape
    if current_shape[0] > target_shape[0]:
        tensor = tensor[:target_shape[0], :]
    if current_shape[1] > target_shape[1]:
        tensor = tensor[:, :target_shape[1]]
    pad_rows = max(0, target_shape[0] - tensor.shape[0])
    pad_cols = max(0, target_shape[1] - tensor.shape[1])
    tensor = F.pad(tensor, (0, pad_cols, 0, pad_rows), value=pad_value)

    return tensor

def get_adj_dist_matrix(mol, symbol_to_index):
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    dist_matrix = Chem.GetDistanceMatrix(mol)
    atom_sequence = [atom.GetSymbol() for atom in mol.GetAtoms()]
    indexed_sequence = [symbol_to_index.get(atom, symbol_to_index["UNK"]) for atom in atom_sequence]

    max_length = 100
    if len(indexed_sequence) < max_length:
        padded_sequence = indexed_sequence + [0] * (max_length - len(indexed_sequence))
    else:
        padded_sequence = indexed_sequence[:max_length]
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
    dist_matrix = torch.tensor(dist_matrix, dtype=torch.float)
    padded_sequence = torch.tensor(padded_sequence, dtype=torch.long)

    return adj_matrix, dist_matrix, padded_sequence

def generate_drug_data_atom(mol_graph, atom_symbols, smiles_rdkit_list, coord, symbol_to_index, smiels):
    atom_dict = {'UNK': 0, 'O': 1, 'Br': 2, 'Al': 3, 'C': 4, 'Mg': 5, 'B': 6, 'I': 7, 'S': 8, 'Li': 9, 'H': 10,
                 'Au': 11, 'F': 12, 'Ca': 13, 'P': 14, 'Na': 15, 'N': 16, 'Bi': 17, 'Cl': 18}

    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features_atom(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features_atom(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T


    gram_matrix = gram_matrix_from_coords(coord)
    gram_matrix = adjust_tensor_size(gram_matrix, (100,100), pad_value=1e9)

    adj_matrix, dist_matrix, pad_atom_seq = get_adj_dist_matrix(mol_graph, atom_dict)
    adj_matrix = adjust_tensor_size(adj_matrix, (100, 100), pad_value=0)
    adj_matrix = (1 - adj_matrix) * (-1e9)  
    dist_matrix = adjust_tensor_size(dist_matrix, (100, 100), pad_value=1e9)

    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index,
                      edge_attr=edge_feats, gram_matrix=gram_matrix,
                      pad_atom_seq=pad_atom_seq, adj_matrix=adj_matrix, dist_matrix=dist_matrix,)

    return data


def generate_drug_data_motif_noglobal(smiles):
    tokenizer = Mol_Tokenizer('token_id.json')
    nums_list, edges, cliques, token_mol_list, token_smiles_list = tokenizer.tokenize(smiles)

    max_length = 70
    if len(nums_list) < max_length:
        padded_sequence = nums_list + [0] * (max_length - len(nums_list))
    else:
        padded_sequence = nums_list[:max_length]
    motif_list_tensor = torch.tensor(nums_list, dtype=torch.long)
    embedding = nn.Embedding(12335, 64)
    embedded_features = embedding(motif_list_tensor)
    padded_sequence = torch.tensor(padded_sequence, dtype=torch.long)

    edges_array = np.array(edges) if edges else np.array([[0, 0]]) 
    edge_index = torch.tensor(edges_array, dtype=torch.long).t().contiguous()    

    data = CustomData(x=embedded_features, edge_index=edge_index, padded_sequence=padded_sequence)

    return data


def load_drug_mol_data():
    

    symbols = list(set(symbols))
    symbol_to_index = {"UNK": 0}
    symbol_to_index.update({symbol: idx + 1 for idx, symbol in enumerate(symbols)})
    for symbol, idx in symbol_to_index.items():
        print(f"{symbol}: {idx}")

    print(f"Length of drug_id_mol_tup: {len(drug_id_mol_tup)}")
    print(f"Length of symbols: {len(symbols)}")

    drug_data_pyg_atom = {
        smiles: generate_drug_data_atom(mol, symbols, smiles_rdkit_list, coord, symbol_to_index, smiles)
        for mol, coord, smiles in tqdm(drug_id_mol_tup, desc='Processing drugs_atom')
    }

    drug_data_pyg_motif_noglobal = {
        smiles: generate_drug_data_motif_noglobal(smiles)
        for mol, coord, smiles in tqdm(drug_id_mol_tup, desc='Processing drugs_motif_feat')
    }


    save_data(drug_data_pyg_atom, 'drug_data_pyg_atom.pkl')
    save_data(drug_data_pyg_motif_noglobal, 'drug_data_pyg_motif.pkl')

def save_data(data, filename):

    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")














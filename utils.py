import os
import torch
import copy
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import xml.etree.ElementTree as ET
import rdkit.Chem as Chem
import json
from rdkit import DataStructs
from rdkit import Geometry
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD

class BestMeter(object):
    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

def load_checkpoint(model_path):
    return torch.load(model_path)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x


class Mol_Tokenizer():
    def __init__(self,tokens_id_file):
        self.vocab = json.load(open(r'{}'.format(tokens_id_file),'r'))
        self.MST_MAX_WEIGHT = 100
        self.get_vocab_size = len(self.vocab.keys())
        self.id_to_token = {value:key for key,value in self.vocab.items()}
    def tokenize(self,smiles):
        mol = Chem.MolFromSmiles(r'{}'.format(smiles))
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        ids,edge = self.tree_decomp(mol)
        motif_list = []
        token_mol_list = []
        token_smiles_list = []
        for id_ in ids:
            token_mol,token_smiles = self.get_clique_mol(mol,id_)
            token_id = self.vocab.get(token_smiles)
            if token_id!=None:
                motif_list.append(token_id)
                token_mol_list.append(token_mol)
                token_smiles_list.append(token_smiles)
            else:
                motif_list.append(self.vocab.get('<unk>'))
                token_mol_list.append(self.vocab.get('<unk>'))
                token_smiles_list.append(self.vocab.get('<unk>'))
        return motif_list, edge, ids, token_mol_list, token_smiles_list
    def sanitize(self,mol):
        try:
            smiles = self.get_smiles(mol)
            mol = self.get_mol(smiles)
        except Exception as e:
            return None
        return mol
    def get_mol(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol
    def get_smiles(self,mol):
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    def get_clique_mol(self,mol,atoms_ids):

        smiles = Chem.MolFragmentToSmiles(mol, atoms_ids, kekuleSmiles=False)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = self.copy_edit_mol(new_mol).GetMol()
        new_mol = self.sanitize(new_mol)  
        return new_mol,smiles
    def copy_atom(self,atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        return new_atom
    def copy_edit_mol(self,mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = self.copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
        return new_mol
    def tree_decomp(self,mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            if len(bonds) > 2 or (len(bonds) == 2 and len(
                    cnei) > 2):  
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:  
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = self.MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)  

        edges = [u + (self.MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        edges = [(row[i], col[i]) for i in range(len(row))]
        return (cliques, edges)


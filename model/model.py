import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter
from torch_geometric.utils import degree, softmax
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn import global_add_pool, global_mean_pool, SAGPooling, global_max_pool, MessagePassing
from layer_graph import DrugEncoder
from layer_motif import Motif
from layer_atom import EncoderModelAtom


class DDI(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_dim = args['hidden_dim']
        in_dim = args['num_atom_type']
        edge_in_dim = args['num_bond_type']
        n_iter = args['n_iter']
        batch_size = args['batch_size']
        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter)
        self.drug_atom_encoder = EncoderModelAtom(4, 256, 8, 128, 0.1, batch_size, use_dist_gram=True)
        self.drug_graph_bipart = Motif(64, 128, 128, 86, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])

        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 9, hidden_dim * 6),
            nn.PReLU(),
            nn.Linear(hidden_dim * 6, hidden_dim * 4),
            nn.PReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, 4)
        )

        self.multi_att = MultiLayerSelfAttention(embed_dim=hidden_dim * 9, num_heads = 8, num_layers= 2)
        self.cross_stitch = CrossStitch(2)

    def forward(self, head_atom, tail_atom, head_motif_noglobal, tail_motif_noglobal, motif_bipart):

        s_h = self.drug_encoder(head_atom)
        s_t = self.drug_encoder(tail_atom)
        h_final_atom = global_add_pool(s_h, head_atom.batch) 
        t_final_atom = global_add_pool(s_t, tail_atom.batch)
        atom_hp = h_final_atom * t_final_atom
        atom_stitch_h, atom_stitch_t = self.cross_stitch([h_final_atom, t_final_atom])
        atom_cat = torch.cat([atom_stitch_h, atom_stitch_t, atom_hp], dim=-1)
        h_graph_bipart, t_graph_bipart = self.drug_graph_bipart(head_motif_noglobal, tail_motif_noglobal, motif_bipart)
        bipart_hp = h_graph_bipart * t_graph_bipart
        bipart_stitch_h, bipart_stitch_t = self.cross_stitch([h_graph_bipart, t_graph_bipart])
        bipart_cat = torch.cat([bipart_stitch_h, bipart_stitch_t, bipart_hp], dim=-1)
        h_atom, _ = self.drug_atom_encoder(head_atom)
        t_atom, _ = self.drug_atom_encoder(tail_atom)
        atompos_hp = h_atom * t_atom
        atompos_stitch_h, atompos_stitch_t = self.cross_stitch([h_atom, t_atom])
        atompos_cat = torch.cat([atompos_stitch_h, atompos_stitch_t, atompos_hp], dim=-1)
        pair = torch.cat([atom_cat, bipart_cat, atompos_cat], dim=-1,)
        pair = self.multi_att(pair)
        logit = self.lin(pair)

        return logit



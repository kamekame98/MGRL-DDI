import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import SAGPooling, global_add_pool, GATConv, LayerNorm, SAGEConv, TransformerConv, GCNConv, Linear


class Motif(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)
        self.initial_node_feature = Linear(64, 64, bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(64)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = Motif_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

    def forward(self, h_data, t_data, b_graph):

        h_data.x = self.initial_node_feature(h_data.x)
        t_data.x = self.initial_node_feature(t_data.x)
        h_data.x = self.initial_node_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_node_norm(t_data.x, t_data.batch)
        h_data.x = F.relu(h_data.x)
        t_data.x = F.relu(t_data.x)

        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        return repr_h, repr_t


class Motif_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads 
        self.in_features = in_features 
        self.out_features = head_out_feats 

        self.feature_conv = TransformerConv(in_features, head_out_feats, n_heads)

        self.intraAtt = IntraGraphAttention(input_dim=128)
        self.interAtt = InterGraphAttention(input_dim=128)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.norm = LayerNorm(n_heads * head_out_feats)

    def forward(self, h_data, t_data, b_graph):

        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)
        h_data.x = self.norm(h_data.x, h_data.batch)
        t_data.x = self.norm(t_data.x, t_data.batch)

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep

        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(h_data.x,
                                                                                                   h_data.edge_index,
                                                                                                   batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout(t_data.x,
                                                                                                   t_data.edge_index,
                                                                                                   batch=t_data.batch)

        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb


class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intra = TransformerConv(input_dim, 32, 2)

    def forward(self, data):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature, edge_index)
        return intra_rep


class InterGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = TransformerConv((input_dim, input_dim), 32, 2)

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])  
        return h_rep, t_rep




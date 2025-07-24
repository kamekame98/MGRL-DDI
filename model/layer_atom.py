
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderModelAtom(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dp_rate, batch_size, use_dist_gram):
        super(EncoderModelAtom, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dp_rate
        self.num_heads = num_heads
        self.use_gram = use_dist_gram
        self.embedding = nn.Embedding(100, self.d_model)
        self.global_embedding = nn.Linear(self.d_model, self.dff)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.d_model, self.num_heads, self.dff, self.dropout_rate) for _ in range(self.num_layers)
        ])
        self.mlp_x = nn.Sequential(
            nn.Linear(256, 128),
            )

    def forward(self, atom):
        x = atom.pad_atom_seq
        batch = atom.batch.max().item() + 1  
        x = x.view(batch, 100)  

        gram_matrix = atom.gram_matrix 

        encoder_padding_mask = create_padding_mask(x) 
        x = self.embedding(x)  

        attention_weights_list_gram = []

        for i in range(self.num_layers):
            x,  attention_weights_gram = self.encoder_layers[i](x, encoder_padding_mask,  gram_matrix)
            attention_weights_list_gram.append(attention_weights_gram)

        x = self.mlp_x(x)

        return x, encoder_padding_mask

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha3 = MultiHeadAttention(d_model, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, encoder_padding_mask, gram_matrix):
        x_gram, attention_weights_gram = self.mha3(x, x, x, encoder_padding_mask, gram_matrix= gram_matrix)
        x_gram = self.dropout1(x_gram)
        out1 = self.layer_norm1(x + x_gram)
        ffn_output = self.mlp(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2, attention_weights_gram

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, gram_matrix):
        batch_size = q.size(0)
        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  
        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, gram_matrix)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)  
        output = self.dense(concat_attention)  

        return output, attention_weights


def create_padding_mask(batch_data):
    padding_mask = (batch_data == 0).float()
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)

    return padding_mask




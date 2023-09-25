import torch
from torch_geometric.nn import  GCNConv
import random

class GCNLinkPredictor(torch.nn.Module):
    """
    GCN based Link Predictor. 
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index, edge_weight=None):
        if edge_weight==None:
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight).relu()
            return self.conv2(x, edge_index, edge_weight)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z, edge_idx, ratio, epoch):
        n_edge = edge_idx.shape[1]
        n_edge_add = int(n_edge*ratio*(epoch-1))

        prob_adj = ((z @ z.t())).sigmoid()
        prob_adj[edge_idx[0,:], edge_idx[1,:]] = -1
        edge_index = (prob_adj > 0.5).nonzero(as_tuple=False).t()
        edge_weight = prob_adj[edge_index[0], edge_index[1]]

        # select top-k edge candidates
        edge_weight_topk = torch.topk(edge_weight, n_edge_add)
        edge_weight_idx = edge_weight_topk.indices
        edge_weight = edge_weight_topk.values
        edge_index = torch.stack((edge_index[0, :][edge_weight_idx], edge_index[1, :][edge_weight_idx]), 0)

        return edge_index, edge_weight

    def merge_edge(self, edge_index, edge_weight, edge_index_add, edge_weight_add):
        edge_index = torch.cat((edge_index, edge_index_add), 1)
        edge_weight = torch.cat((edge_weight, edge_weight_add))

        return edge_index, edge_weight


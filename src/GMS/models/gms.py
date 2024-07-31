"""
GNN for mixture similarity (GMS). This is a graph iso morphism network that
is tailored to identify how similar two mixtures of compounds are for humans.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class GMS(torch.nn.Module):
    """
    model instantiation
    """
    def __init__(self, config):
        super().__init__()

        # setting dims with config
        node_dim = config["data"]["node_feature_len"]
        edge_dim = config["data"]["edge_feature_len"]
        emb_hidden = config["model_params"]["emb_hidden"]
        emb_out = config["model_params"]["emb_out"]
        GINE_hidden = config["model_params"]["GINE_hidden"]
        GINE_out = config["model_params"]["GINE_out"]
        proj_hidden = config["model_params"]["proj_hidden"]
        proj_out = config["model_params"]["proj_out"]

        self.mp_layers = config["model_params"]["mp_layers"]
        self.p = config["model_params"]["dropout"]

        # embed nodes and edges
        self.emb_node = nn.Sequential(
            nn.Linear(node_dim, emb_hidden),
            nn.ReLU(),
            nn.Linear(emb_hidden, emb_out)
        )
        self.emb_edges = nn.Sequential(
            nn.Linear(edge_dim, emb_hidden),
            nn.ReLU(),
            nn.Linear(emb_hidden, emb_out)
        )

        # Messagepassing layer
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(emb_out, GINE_hidden),
                nn.ReLU(),
                nn.Linear(GINE_hidden, GINE_out)
            ),
            train_eps=True,
        )
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(GINE_out, GINE_hidden),
                nn.ReLU(),
                nn.Linear(GINE_hidden, GINE_out)
            ),
            train_eps=True
        )

        # projection
        self.proj = nn.Sequential(
            nn.Linear(GINE_out, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_out)
        )

    def forward(self, data):
        x, e, e_idx = data.x, data.edge_attr, data.edge_index

        # embed nodes and edges
        x = self.emb_node(x)
        e = self.emb_edges(e)

        # Message passing
        x = self.conv1(x, e_idx, e)
        x = F.dropout(x, p=self.p, training=self.training)
        for i in range(self.mp_layers - 1):
            x = self.conv2(x, e_idx, e)
            x = F.dropout(x, p=self.p, training=self.training)

        # global pooling
        g = global_mean_pool(x, batch=data.mix_batch)
        g = self.proj(g)

        return g


if __name__ == '__main__':
    pass

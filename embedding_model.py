import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from loss import EmbeddingLossFunctions
from tqdm import tqdm

class EmbModel(nn.Module):
    def __init__(self, n_nodes, embedding_dim):
        super(EmbModel, self).__init__()
        self.node_embedding = nn.Embedding(n_nodes, embedding_dim)
        self.n_nodes = n_nodes
        self.link_pred_layer = EmbeddingLossFunctions()

    def forward(self, nodes):
        node_output = self.node_embedding(nodes)
        node_output = F.normalize(node_output, dim=1)
        return node_output

    def edge_loss(self, edges, neg):
        source_embedding = self.forward(edges[:, 0])
        target_embedding = self.forward(edges[:, 1])
        neg_embedding = self.forward(neg)
        loss = self.link_pred_layer.loss(source_embedding, target_embedding, neg_embedding)
        loss = loss/len(edges)
        return loss


    def hyperedge_loss(self, checkins, neg_checkins):
        checkin_embs = self.forward(checkins) # 8 x 4 x 128
        checkin_emb_means = F.normalize(torch.mean(checkin_embs, dim=1), dim=1) # 8 x 128
        neg_embs = self.forward(neg_checkins)
        loss = 0
        for i in range(checkins.shape[1]):
            loss += self.link_pred_layer.loss(checkin_emb_means, checkin_embs[:, i, :], neg_embs[:, i, :]) / len(checkins)
        return loss / 4

    def hyperedge_loss2(self, Nodes, negs):
        user_embs = self.forward(Nodes[0])
        time_embs = self.forward(Nodes[1])
        location_embs = self.forward(Nodes[2])
        cate_embs = self.forward(Nodes[3])

        user_means = torch.mean(user_embs, dim=0)
        time_means = torch.mean(time_embs, dim=0)
        loc_means = torch.mean(location_embs, dim=0)
        cate_means = torch.mean(cate_embs, dim=0)

        user_means = user_means.repeat(len(user_embs), 1)
        time_means = time_means.repeat(len(time_embs), 1)
        loc_means = loc_means.repeat(len(location_embs), 1)
        cate_means = cate_means.repeat(len(cate_embs), 1)

        neg_users = self.forward(negs[0])
        neg_times = self.forward(negs[1])
        neg_locs = self.forward(negs[2])
        neg_cates = self.forward(negs[3])

        loss0, _, _ = self.link_pred_layer.loss(user_means, user_embs, neg_users)
        loss1, _, _ = self.link_pred_layer.loss(time_means, time_embs, neg_times)
        loss2, _, _ = self.link_pred_layer.loss(loc_means, location_embs, neg_locs)
        loss3, _, _ = self.link_pred_layer.loss(cate_means, cate_embs, neg_cates)

        loss = loss0 + loss1 + loss2 + loss3
        loss = loss / len(user_embs)
        return loss



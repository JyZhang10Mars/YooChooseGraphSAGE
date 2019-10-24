#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 2019/8/13 15:54
# Email：jyzhang.mars@gmail.com
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv, SGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class CLICK_Net(torch.nn.Module):
    def __init__(self, embed_dim, df):
        super(CLICK_Net, self).__init__()

        self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() + 1, embedding_dim=embed_dim)

        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)   # 初始化节点embedding
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))   # SAGEConv卷积
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)   # TopK池化 保留 kN 个节点(k=ratio) 类似于Dropout
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # shape:[*, 128x2=256]

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3  # shape:[*, 128x2=256]

        x = self.lin1(x)  # shape:[*, 128]
        x = self.act1(x)
        x = self.lin2(x)  # shape:[*, 64]
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # shape:[*, 1]

        return x


class BUY_Net(torch.nn.Module):
    def __init__(self, click_df, EMBED_DIM):
        super(BUY_Net, self).__init__()

        self.item_embedding = torch.nn.Embedding(num_embeddings=click_df.item_id.max() + 1, embedding_dim=EMBED_DIM)
        self.category_embedding = torch.nn.Embedding(num_embeddings=click_df.category.max() + 2, embedding_dim=EMBED_DIM)

        self.conv1 = GraphConv(EMBED_DIM * 2, 128)
        self.pool1 = TopKPooling(128, ratio=0.9)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.9)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.9)

        self.lin1 = torch.nn.Linear(256, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        item_id = x[:, :, 0]
        category = x[:, :, 1]

        emb_item = self.item_embedding(item_id).squeeze(1)
        emb_category = self.category_embedding(category).squeeze(1)

        #            emb_item = emb_item.squeeze(1)
        #            emb_cat
        x = torch.cat([emb_item, emb_category], dim=1)
        #             print(x.shape)
        x = F.relu(self.conv1(x, edge_index))
        #             print(x.shape)
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.act2(x)

        outputs = []
        for i in range(x.size(0)):
            output = torch.matmul(emb_item[data.batch == i], x[i, :])

            outputs.append(output)

        x = torch.cat(outputs, dim=0)
        x = torch.sigmoid(x)

        return x
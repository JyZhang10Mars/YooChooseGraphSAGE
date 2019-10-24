#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 2019/8/13 15:54
# Email：jyzhang.mars@gmail.com
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class YooChooseClickDataset(InMemoryDataset):

    def __init__(self, root, df, transform=None, pre_transform=None):
        super(YooChooseClickDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.df = df

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["yoochoose_click_binary.dataset"]

    def download(self):
        pass

    def process(self):

        data_list = []

        grouped = self.df.groupby("session_id")  # 按session_id进行分组

        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)  # 对item_id再次标签编码
            group = group.reset_index(drop=True)
            group["sess_item_id"] = sess_item_id

            # 构建节点特征（item_id）
            node_features = group.loc[group.session_id == session_id, ["sess_item_id", "item_id"]].sort_values(
                "sess_item_id").item_id.drop_duplicates().values
            node_features = torch.LongTensor(node_features).unsqueeze(1)

            # 构建边索引
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            x = node_features
            y = torch.FloatTensor([group.label.values[0]])

            # 构建图数据对象
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class YooChooseBuyDataset(InMemoryDataset):

    def __init__(self, root, click_df, buy_item_dict, transform=None, pre_transform=None):
        super(YooChooseBuyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.click_df = click_df
        self.buy_item_dict = buy_item_dict

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["yoochoose_buy_binary.dataset"]

    def download(self):
        pass

    def process(self):

        data_list = []
        # process by session_id
        grouped = self.click_df.groupby("session_id")

        for session_id, group in tqdm(grouped):

            le = LabelEncoder()
            sess_item_id = le.fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group["sess_item_id"] = sess_item_id

            node_features = \
            group.loc[group.session_id == session_id, ["sess_item_id", "item_id", "category"]].sort_values(
                "sess_item_id")[["item_id", "category"]].drop_duplicates().values
            node_features = torch.LongTensor(node_features).unsqueeze(1)  # shape：[*, 1, 2]   [item, 0]

            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            x = node_features

            # 构建label，假如一个session_id有3个item_id，则[1, 0, 0]代表购买了item_id=0的物品
            if session_id in self.buy_item_dict:
                positive_indices = le.transform(self.buy_item_dict[session_id])
                label = np.zeros(len(node_features))
                label[positive_indices] = 1
            else:
                label = [0] * len(node_features)

            y = torch.FloatTensor(label)

            data = Data(x=x, edge_index=edge_index, y=y)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
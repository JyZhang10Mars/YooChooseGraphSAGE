#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 2019/9/19 10:24
# Email：jyzhang.mars@gmail.com
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataset import YooChooseBuyDataset
from model import BUY_Net

torch.manual_seed(42)



def train():
    model.train()

    loss_all = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return roc_auc_score(labels, predictions)

############################################
#               Preprocessing              #
############################################
print("Preprocessing...")


# 加载数据
click_df = pd.read_csv("./input/yoochoose-clicks.dat", header=None)  # 33003944
click_df.columns = ["session_id", "timestamp", "item_id", "category"]
buy_df = pd.read_csv("./input/yoochoose-buys.dat", header=None)  # 1150753
buy_df.columns = ["session_id", "timestamp", "item_id", "price", "quantity"]


# 过滤掉item_id小于2的session，map()会根据提供的函数对指定序列做映射
click_df["valid_session"] = click_df.session_id.map(click_df.groupby("session_id")["item_id"].size() > 2)
click_df = click_df.loc[click_df.valid_session].drop("valid_session", axis=1)


# 由于数据非常大，对其进行二次采样
sampled_session_id = np.random.choice(click_df.session_id.unique(), 1000000, replace=False)
click_df = click_df.loc[click_df.session_id.isin(sampled_session_id)]
click_df.nunique()


# session 中 item_id（节点） 的平均数量
# click_df.groupby("session_id")["item_id"].size().mean()


# 标签编码
item_encoder = LabelEncoder()
category_encoder = LabelEncoder()
click_df["item_id"] = item_encoder.fit_transform(click_df.item_id)
click_df["category"] = category_encoder.fit_transform(click_df.category.apply(str))


# 保持buy_df与click_df中的session_id一致
buy_df = buy_df.loc[buy_df.session_id.isin(click_df.session_id)]
buy_df["item_id"] = item_encoder.transform(buy_df.item_id)


# 构建session_id-item_id字典，key：session_id，value：list of item_id
buy_item_dict = dict(buy_df.groupby("session_id")["item_id"].apply(list))


# 划分数据集
dataset = YooChooseBuyDataset(root="./", click_df=click_df, buy_item_dict=buy_item_dict)
dataset = dataset.shuffle()
one_tenth_length = int(len(dataset) * 0.1)
train_dataset = dataset[:one_tenth_length * 8]
val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
test_dataset = dataset[one_tenth_length * 9:]
print("Len of Train, Val, Test set:", len(train_dataset), len(val_dataset), len(test_dataset))


# DataLoader加载数据
BATCH_SIZE = 512
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
num_items = click_df.item_id.max() + 1
num_categories = click_df.category.max() + 1


############################################
#           Training & Testing             #
############################################
print("Training & Testing...")

# 构建模型
LR = 0.001
EMBED_DIM = 128
NUM_EPOCHS = 10
device = torch.device("cuda")
model = BUY_Net(click_df=click_df, EMBED_DIM=EMBED_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
crit = torch.nn.BCELoss()


# 训练 & 测试
print("{} Epoches".format(NUM_EPOCHS))
for epoch in range(1, NUM_EPOCHS+1):
    print("Epoch: {:03d}".format(epoch))
    loss = train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    print("Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}".format(loss, train_acc, val_acc, test_acc))

print("Done!")
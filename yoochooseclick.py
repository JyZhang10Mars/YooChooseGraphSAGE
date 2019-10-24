#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 2019/8/13 15:
# Email：jyzhang.mars@gmail.com
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from dataset import YooChooseClickDataset
from model import CLICK_Net
from evaluate import evaluate

############################################
#               Preprocessing              #
############################################
print("Preprocessing...")

# 载入数据
click_df = pd.read_csv("./input/yoochoose-clicks.dat", header=None)  # click_df：33003944
click_df.columns = ["session_id", "timestamp", "item_id", "category"]

buy_df = pd.read_csv("./input/yoochoose-buys.dat", header=None)  # buy_df：1150753
buy_df.columns = ["session_id", "timestamp", "item_id", "price", "quantity"]


# 对"item_id"进行标签编码
item_encoder = LabelEncoder()
click_df["item_id"] = item_encoder.fit_transform(click_df.item_id)


# 由于数据较大，对其进行二次采样
sampled_session_id = np.random.choice(click_df.session_id.unique(), 1000000, replace=False)
click_df = click_df.loc[click_df.session_id.isin(sampled_session_id)]


# 确定click事件的label（即是否购买）
click_df["label"] = click_df.session_id.isin(buy_df.session_id)


# 训练，验证和测试集划分
dataset = YooChooseClickDataset(root="./", df=click_df)
dataset = dataset.shuffle()
train_dataset = dataset[:800000]
val_dataset = dataset[800000:900000]
test_dataset = dataset[900000:]
print("Len of Train, Val, Test set:", len(train_dataset), len(val_dataset), len(test_dataset))

# DataLoader加载数据
BATCH_SIZE = 1024
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


############################################
#                 Training                 #
############################################
print("Training...")

# 构建模型
EMBED_DIM = 128
NUM_EPOCHS = 10
LR = 0.001
device = torch.device("cuda")
model = CLICK_Net(EMBED_DIM, click_df).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
crit = torch.nn.BCELoss()


# 训练
print("{} Epoches".format(NUM_EPOCHS))
for epoch in range(NUM_EPOCHS):
    model.train()
    print("Epoch-{:02d}：".format(epoch + 1))
    loss_all = 0
    for data in tqdm(train_loader):
        # to gpu
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

        train_loss = loss_all / len(train_dataset)

    print("Loss：{}".format(train_loss))


# 保存模型
# torch.save(model, "./")
# net = torch.load("./")


############################################
#                 Testing                  #
############################################
print("Testing...")

# 测试
for epoch in range(1):
    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)
    test_acc = evaluate(model, test_loader)
    print("Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}".format(train_acc, val_acc, test_acc))

print("Done!")
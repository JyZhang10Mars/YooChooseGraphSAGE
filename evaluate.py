#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 2019/8/13 16:54
# Email：jyzhang.mars@gmail.com
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def evaluate(model, data_loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            device = torch.device("cuda")
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return roc_auc_score(labels, predictions)

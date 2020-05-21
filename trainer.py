import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc; import os
import torch
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F

from transformer import *
from args import args

X = Transformer(
    n_token=args["n_token"],
    n_layer=args["n_layer"],
    n_head=args["n_head"],
    d_model=args["d_model"],
    d_head=args["d_head"],
    d_inner=args["d_inner"],
    dropout=args["dropout"],
    dropatt=args["dropatt"],
    dtype=torch.float32,
    attention_dropout_prob=args["attention_dropout_prob"],
    output_dropout_prob=args["output_dropout_prob"],
    init_method=torch.optim.SGD,
    bi_data=10
)

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(X.parameters(), lr=0.001, momentum=0.9)

from sklearn.model_selection import train_test_split as T
train = pd.read_csv('training.csv.zip')
train = train.drop(['id'], axis=1)
train = train.drop(['item_id'], axis=1)
train = train.drop(['dept_id'], axis=1)
train = train.drop(['cat_id'], axis=1)
train = train.drop(['store_id'], axis=1)
train = train.drop(['state_id'], axis=1)
X_train, y_train = T(train, test_size=0.1)

#X_train = X_train.head(28)
#X_train = torch.LongTensor(X_train.values)
print(X_train.size(0))
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    gc.collect()
    for i, data in enumerate(train.head(1500), 0):
        X.train()      
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = X_train, y_train
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = X(inputs, torch.LongTensor(train.head(28).values.astype("float64")), crit=None, mems=None)
        gc.collect()
        outputs = torch.FloatTensor(outputs)
        outputs.requires_grad = True
        loss = criterion(outputs.view(-1).reshape(14336, 1939), torch.FloatTensor(train.head(14336).values.astype("float64")))
        gc.collect()
        loss.backward()
        optimizer.step()
        gc.collect()

        # print statistics
        running_loss += loss.item()
        HIST = []
        if i % 28 == 0:    
            !nvidia-smi
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            # average across batches
            HIST.append(outputs)
            gc.collect()
            running_loss = 0.0

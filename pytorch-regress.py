import numpy as np
import torch

###=============================================
### 自定义线性回归
###=============================================
#inputs
inputs = np.array([[73,67,43],
                   [91,88,64],
                   [87,134,58],
                   [102,43,37],
                   [69,96,70]],dtype='float32')

##Targets(apple,oranges)
targets = np.array([[56,70],
                   [81,101],
                   [119,133],
                   [22,37],
                   [103,119]],dtype='float32')
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
# print(inputs)
# print(targets)
#
# w = torch.randn(2,3,requires_grad=True)
# b = torch.randn(2,requires_grad = True)
# print(w)
# print(b)
#
# def model(x):
#     return x @ w.t() + b
# # pred = model(inputs)
# # print('pred:\n',pred)
# # print('targets\n:',targets)
#
# def mse(pred,targets):
#     diff = pred - targets
#     return torch.sum(diff*diff) / diff.numel()
# # loss = mse(pred,targets)
# # print('loss:',loss)
# # loss.backward()
#
# for i in range(100):
#     pred = model(inputs)
#     loss = mse(pred, targets)
#     loss.backward()
#     with torch.no_grad():
#         w -= w.grad * 1e-5
#         b -= b.grad * 1e-5
#         w.grad.zero_()
#         b.grad.zero_()
# pred = model(inputs)
# loss = mse(pred,targets)
# print('loss:',loss)
# print('pred:\n',pred)
# print('targets\n:',targets)


###=============================================
### pytorch 线性回归
###=============================================

import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


batch_size = 5
train_ds = TensorDataset(inputs,targets)
train_dl = DataLoader(train_ds,batch_size,shuffle = True)

model = nn.Linear(3,2)
print('weight:\n',model.weight)
print('bias:\n',model.bias)
loss_fn = F.mse_loss
loss = loss_fn(model(inputs),targets)
print('loss:\n',loss)
opt = torch.optim.SGD(model.parameters(),lr=1e-5)

pred = model(inputs)
print('pred:\n',pred)


# for epoch in (100):
#     model.train()
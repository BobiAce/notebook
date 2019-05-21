#coding utf-8
import numpy as np
import torch
##==========================================
##   softmax公式
# def softmax(x):
#     """Compute the softmax of vector x."""
#
#     # exp_x = np.exp(x)
#     # softmax_x = exp_x / np.sum(exp_x)
#     # x = np.array(x)
#     print(x.ndim)
#     exp_x = np.exp(x)
#     ##这里的x传入是(n,c),所以你求sum的时候千万不能直接求sum，要对axis=1求和，axis：沿着横轴操作
#     sum = np.sum(exp_x,axis=1).reshape(x.shape[0],1)
#     softmax_x = exp_x / sum
#
#     return softmax_x
# if __name__ == '__main__':
#     x = np.array([[1, 2, 1, 1],
#                   [2, 3, 1, 1],
#                   [3, 4, 1, 1]])
#     soft = softmax(x)
#     print(soft)



##==========================================
##   softmax回归 梯度公式 推导及实现

x = np.array([[1, 2, 1, 0],
              [2, 3, 1, 1],
              [3, 4, 1, 2]])
label = np.array([[0],[1],[0],[1]])

# theta = np.array([[1, 1, 0,1], [0, 1, 1,0]])
theta=np.random.randn(2,4)#初始theta矩阵  










# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler  # 引入归一化的包

###=============================================
###  线性回归  sklearn.linear_model求解,X是二维
# def linearRegression():
#     print(u"加载数据...\n")
#     data = loadtxtAndcsv_data("data.txt", ",", np.float64)  # 读取数据
#     X = np.array(data[:, 0:-1], dtype=np.float64)  # X对应0到倒数第2列
#     y = np.array(data[:, -1], dtype=np.float64)  # y对应最后一列
#
#     # 归一化操作
#     scaler = StandardScaler()
#     scaler.fit(X)
#     x_train = scaler.transform(X)
#     # x_test = scaler.transform(np.array([10, 3]))
#     x_test = np.array(data[0:2,0:-1],dtype=np.float64)
#     x_test = scaler.transform(x_test)
#     # 线性模型拟合
#     model = linear_model.LinearRegression()
#     model.fit(x_train, y)
#
#     # 预测结果
#     result = model.predict(x_test)
#     print(model.coef_)  # Coefficient of the features 决策函数中的特征系数
#     print(model.intercept_)  # 又名bias偏置,若设置为False，则为0
#     print(result)  # 预测结果
#
#
# # 加载txt和csv文件
# def loadtxtAndcsv_data(fileName, split, dataType):
#     return np.loadtxt(fileName, delimiter=split, dtype=dataType)
#
#
# # 加载npy文件
# def loadnpy_data(fileName):
#     return np.load(fileName)
#
# if __name__ == "__main__":
#     linearRegression()







###=============================================
###  线性回归  sklearn.linear_model求解 X是一维
###    最小二乘法一般是解线性方程组的一种方法，多少个方程多少个
###     未知数
'''
下面代码实现的是梯度下降法线性拟合，并且包含自己造的轮子与调用库的结果比较。
问题：对直线附近的带有噪声的数据进行线性拟合，最终求出w,b的估计值。
代码中self_func()函数为自定义拟合函数，skl_func()为调用scikit-learn中线性模块的函数。
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

n = 101

x = np.linspace(0, 10, n)
ppp = x.reshape(-1, 1)
noise = np.random.randn(n)
y = 2.5 * x + 0.8 + 2.0 * noise

def self_func(steps=100, alpha=0.01):
    w = 0.5
    b = 0
    alpha = alpha
    for i in range(steps):
        y_hat = w * x + b
        dy = 2.0 * (y_hat - y)
        dw = dy * x
        db = dy
        w = w - alpha * np.sum(dw) / n
        b = b - alpha * np.sum(db) / n
        e = np.sum((y_hat - y) ** 2) / n
        print (i,'W=',w,'\tb=',b,'\te=',e)
    print('self_func:\tW =', w, '\n\tb =', b)
    plt.scatter(x, y)
    plt.plot(np.arange(0, 10, 1), w * np.arange(0, 10, 1) + b, color='r', marker='o',
             label='self_func(steps=' + str(steps) + ', alpha=' + str(alpha) + ')')


def skl_func():
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    y_hat = lr.predict(np.arange(0, 10, 0.75).reshape(-1, 1))
    print('skl_fun:\tW = %f\n\tb = %f' % (lr.coef_, lr.intercept_))
    plt.plot(np.arange(0, 10, 0.75), y_hat, color='g', marker='x', label='skl_func')

self_func(10000)
skl_func()
plt.legend(loc='upper left')
plt.show()

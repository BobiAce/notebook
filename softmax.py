#coding utf-8
import numpy as np

##==========================================
##   softmax公式
# def softmax(x):
#     """Compute the softmax of vector x."""
#
#     # exp_x = np.exp(x)
#     # softmax_x = exp_x / np.sum(exp_x)
#     x = np.array(x)
#     x = x - np.max(x)
#     exp_x = np.exp(x)
#     softmax_x = exp_x / np.sum(exp_x)
#     return softmax_x
# if __name__ == '__main__':
#     x = [1000,2000,3000]
#     soft = softmax(x)
#     print(soft)



##==========================================
##   softmax回归 梯度公式 推导及实现

#采用随机梯度下降法，每次只挑选一个样本做优化
#采用随机梯度下降法，每次只挑选一个样本做优化
def softMax(x,y,alpha):
    theta=np.zeros((3,5))#初始theta矩阵  
    for i in range(10000): #迭代10000次  
        k=np.random.randint(0,105) #从105个样本中随机挑选一个做优化
        x_=x[k].reshape(5,1)
        theta_T_x=np.dot(theta,x_)#计算所有的theta*x  
        e_theta_T_x=np.exp(theta_T_x)#计算所有指数函数  
        denominator=e_theta_T_x.sum()#计算分母  
        numerator=e_theta_T_x        #分子  
        fraction=numerator/denominator#计算所有分数  
        y_vector=np.where(np.arange(3).reshape(3,1)==y[k],1,0) #计算y向量
        gradient=(fraction-y_vector)*x[k]
        theta-=alpha*gradient#更新theta矩阵
    return theta
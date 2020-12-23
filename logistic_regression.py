#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

def loaddata(file, delimiter):
    data = np.loadtxt(file, delimiter=delimiter)
    print('Dimesions: ', data.shape)
    print(data[1:6,:])
    return (data)

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:,2] == 0
    pos = data[:,2] == 1

    if axes == None:
        axes = plt.gca()

    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True)

data = loaddata('data1.txt', ',')

X = np.c_[np.ones((data.shape[0], 1)), data[:,0:2]]
y = np.c_[data[:,2]]

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
# 画出所有点
#plt.show()

def sigmoid(z):
    return (1/(1+np.exp(-z)))


# 定义损失函数 
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))

    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))

    if np.isnan(J[0]):
        return np.inf
    return J[0]


# 求解梯度
def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))

    grad =(1.0/m)*X.T.dot(h-y)

    return grad.flatten()

initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)

#Cost:
# 0.6931471805599452
#Grad:
# [ -0.1        -12.00921659 -11.26284221]


# 最小化损失函数
res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient, options={'maxiter':400})


print('----minimize cost function----')
print(res)

# 预测
def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

# 考试1得分45， 考试2得分85的同学通过概率多高
p = sigmoid(np.array([1, 45, 85]).dot(res.x.T))
print(p)

## 画决策边界
#plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
#plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
#x1_min, x1_max = X[:,1].min(), X[:,1].max(),
#x2_min, x2_max = X[:,2].min(), X[:,2].max(),
#xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
#h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
#h = h.reshape(xx1.shape)
#plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
#plt.show()


















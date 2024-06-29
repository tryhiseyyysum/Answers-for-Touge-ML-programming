#encoding=utf8 
import numpy as np
from numpy.linalg import inv
def lda(X, y):
    '''
    input:X(ndarray):待处理数据
          y(ndarray):待处理数据标签，标签分别为0和1
    output:X_new(ndarray):处理后的数据
    '''
    #********* Begin *********#
    # 划分出第一类样本与第二类样本
    x0 = X[y == 0]
    x1 = X[y == 1]
    
    # 获取第一类样本与第二类样本中心点
    mean0 = np.mean(x0, axis=0)
    mean1 = np.mean(x1, axis=0)
    
    # 计算第一类样本与第二类样本协方差矩阵
    cov0 = np.dot((x0-mean0).T,(x0-mean0))
    cov1 = np.dot((x1-mean1).T,(x1-mean1))
    
    # 计算类内散度矩阵
    S_w = cov0 + cov1
    
    # 计算w
    w = inv(S_w).dot((mean0 - mean1).reshape(len(mean0),1))
    
    # 计算新样本集
    X_new = X.dot(w)
    #********* End *********#
    return X_new


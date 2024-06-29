import numpy as np
def lle(data,d,k):
    '''
    input:data(ndarray):待降维数据,行数为样本个数，列数为特征数
          d(int):降维后数据维数
          k(int):最近的k个样本
    output:Z(ndarray):降维后的数据
    '''
    #********* Begin *********#
    #确定样本i的邻域
    m,n = data.shape
    w = np.zeros((m,m))
    M = np.zeros((m,m))
    for i in range(m):      # 遍历每个样本
        c = np.zeros((k,k))     #初始化矩阵c
        distance = np.square(data[i]-data).sum(axis=1)  #计算样本i到其他样本的距离平方
        index = np.argsort(distance)    #对距离进行排序,返回的是索引,升序
        q = index[1:k+1]        #取最近的k个样本，就是邻居，从1开始是因为第一个是自己
        #求矩阵c及其逆
        for l in range(len(q)):
            for s in range(len(q)):
                c[l][s] = np.dot(data[i]-data[q[l]],(data[i]-data[q[s]]).T)
        c_ = np.linalg.inv(c)
        down = np.sum(c_)
        #求w
        for l in range(len(q)):
            up = np.sum(c_[l])
            w[i,q[l]] = up/down
    #求得M并矩阵分解
    M = np.dot((np.eye(m)-w).T,np.eye(m)-w)
    lamda,V=np.linalg.eigh(M)  #返回按特征值大小升序的特征值和特征向量
    #求Z
    Z = V[:,1:1+d]      #取第 2 小到第 d+1 小的特征值所对应的特征向量组成的矩阵，因为最小的特征值向量非常接近 0，一般从第二小的开始取
    #********* End *********#
    return Z   
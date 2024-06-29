import numpy as np
def mds(data,d):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维数
    output:Z(ndarray):降维后的数据
    '''
    #********* Begin *********#
    #计算dist2,dist2i,dist2j,dist2ij
    m,n = data.shape
    dist =np.zeros((m,m))
    disti = np.zeros(m)
    distj = np.zeros(m)
    B = np.zeros((m,m))

    # 计算dist2
    for i in range(m):
        dist[i] = np.sum(np.square(data[i]-data),axis=1).reshape(1,m)  # 计算dist[i,j]的平方,即所有样本两两之间的距离

    # 计算dist2i,dist2j,dist2ij
    for i in range(m):
        disti[i] = np.mean(dist[i,:])   # 计算dist2i
        distj[i] = np.mean(dist[:,i])   # 计算dist2j

    distij = np.mean(dist)   # 对整个矩阵求均值，得到dist2..

    #计算B
    for i in range(m):
        for j in range(m):            
            B[i,j] = -0.5*(dist[i,j] - disti[i] - distj[j] + distij)

    #矩阵分解得到特征值与特征向量
    lamda,V=np.linalg.eigh(B)   #eigh函数返回的特征值是升序排列的，eigh函数只适用于对称矩阵，处理起来比eig函数更快
    #计算Z
    index=np.argsort(-lamda)[:d]    #先降序，再取最大的d个特征值对应的特征向量
    diag_lamda=np.sqrt(np.diag(-np.sort(-lamda)[:d]))  #np.diag()将一维数组转换为对角矩阵
    V_selected=V[:,index]    #前d个最大特征值对应的特征向量
    Z=V_selected.dot(diag_lamda)
    #********* End *********#
    return Z 
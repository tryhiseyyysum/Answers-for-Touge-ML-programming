import numpy as np
def isomap(data,d,k,Max=10000):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维数
          k(int):最近的k个样本
          Max(int):表示无穷大
    output:Z(ndarray):降维后的数据
    '''
    #********* Begin *********#
    #计算dist2,dist2i,dist2j,dist2ij
    m,n = data.shape
    dist = np.ones((m,m))*Max
    disti = np.zeros(m)
    distj = np.zeros(m)
    B = np.zeros((m,m))

    #在距离计算上，与MDS的不同在于，这里是计算样本i到邻居l的距离，每次都只计算与邻居的距离，从而扩展到全局，得到全部距离
    #而MDS对于两两之间的样本，不考虑邻居，都统一通过欧氏距离进行计算
    for i in range(m):
        distance = np.square(data[i]-data).sum(axis=1)  #计算样本i到其他样本的距离
        index = np.argsort(distance)     #对距离进行排序,返回的是索引,升序
        q = index[:k]       #取最近的k个样本，就是邻居
        for l in q:     #遍历邻居
            dist[i][l] = np.square(data[i]-data[l]).sum()  #计算样本i到邻居l的距离
    for i in range(m):
        disti[i] = np.mean(dist[i,:])
        distj[i] = np.mean(dist[:,i])
    distij = np.mean(dist)        
    #计算B
    for i in range(m):
        for j in range(m):            
            B[i,j] = -0.5*(dist[i,j] - disti[i] - distj[j] + distij)
    #矩阵分解得到特征值与特征向量
    lamda,V=np.linalg.eigh(B)
    #计算Z
    index=np.argsort(-lamda)[:d]
    diag_lamda=np.sqrt(np.diag(-np.sort(-lamda)[:d]))
    V_selected=V[:,index]
    Z=V_selected.dot(diag_lamda)
    #********* End *********#
    return Z
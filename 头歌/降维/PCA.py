import numpy as np

def pca(data, k):
    '''
    对data进行PCA，并将结果返回
    :param data:数据集，类型为ndarray
    :param k:想要降成几维，类型为int
    :return: 降维后的数据，类型为ndarray
    '''

    #********* Begin *********#
    u=np.mean(data,axis=0)   #所有样本在同一维度求均值，最终得到的维度是特征的维度
    data-=u  #零均值化

    # data的行数为样本个数，列数为特征个数
    # 由于cov函数的输入希望是行代表特征，列代表数据的矩阵，所以要转置
    cov=np.cov(data.T)   #求协方差矩阵
    value,vector=np.linalg.eig(cov)  #特征值分解
    
    index=np.argsort(-value)   #对特征值进行排序,降序，加个负号再进行默认的升序排序最终就是降序，返回元素值降序的索引
    vector=vector[:,index]   #对特征向量进行排序，排在前面的是最大的特征值对应的特征向量
    data_pca=np.dot(data,vector[:,:k])  #降维，前k个最大特征值对应的特征向量组成了投影矩阵
    return data_pca

    #********* End *********#
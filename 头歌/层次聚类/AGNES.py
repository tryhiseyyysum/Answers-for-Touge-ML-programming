import numpy as np

def calc_max_dist(cluster1, cluster2):
    '''
    计算簇间最大距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的最大距离
    '''
    max_dist=-np.inf
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            dist=np.linalg.norm(cluster1[i]-cluster2[j])
            if dist>max_dist:
                max_dist=dist
    return max_dist


def AGNES(feature, k):
    '''
    AGNES聚类并返回聚类结果，量化距离时请使用簇间最大欧氏距离
    假设数据集为`[1, 2], [10, 11], [1, 3]]，那么聚类结果可能为`[[1, 2], [1, 3]], [[10, 11]]]
    :param feature:数据集，类型为ndarray
    :param k:表示想要将数据聚成`k`类，类型为`int`
    :return:聚类结果，类型为list
    '''

    #********* Begin *********#
    C=[]
    for d in feature:
        C.append(np.array([d]))     #初始化：每个样本作为一个簇


    q=len(C)
    while q>k:
        min_dist=np.inf
        for i in range(len(C)):
            for j in range(i+1,len(C)):
                dist=calc_max_dist(C[i],C[j])
                if dist<min_dist:
                    min_dist=dist
                    merge_i=i
                    merge_j=j
        C[merge_i]=np.vstack((C[merge_i],C[merge_j]))  #合并簇，竖直拼接是因为一行代表一个样本
        C.pop(merge_j)
        q-=1
    return C
    #********* End *********#

#测试
feature = np.array([[1, 2], [10, 11], [1, 3]])
k = 2
print(AGNES(feature, k))
#输出[[array([1, 2]), array([1, 3])], [array([10, 11])]]

import numpy as np


def calc_min_dist(cluster1, cluster2):
    '''
    计算簇间最小距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的最小距离
    '''

    #********* Begin *********#
    min_dist=np.inf
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            dist=np.linalg.norm(cluster1[i]-cluster2[j])
            if dist<min_dist:
                min_dist=dist
    return min_dist
    #********* End *********#


def calc_max_dist(cluster1, cluster2):
    '''
    计算簇间最大距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的最大距离
    '''

    #********* Begin *********#
    max_dist=-np.inf
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            dist=np.linalg.norm(cluster1[i]-cluster2[j])
            if dist>max_dist:
                max_dist=dist
    return max_dist
    #********* End *********#


def calc_avg_dist(cluster1, cluster2):
    '''
    计算簇间平均距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的平均距离
    '''

    #********* Begin *********#
    sum_dist=0
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            sum_dist+=np.linalg.norm(cluster1[i]-cluster2[j])
    return sum_dist / (len(cluster1)*len(cluster2))
    #********* End *********#
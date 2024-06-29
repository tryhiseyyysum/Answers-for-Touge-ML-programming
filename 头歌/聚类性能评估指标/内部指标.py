import numpy as np
def calc_DBI(feature, pred):
    '''
    计算并返回DB指数
    :param feature: 待聚类数据的特征，类型为`ndarray`
    :param pred: 聚类后数据所对应的簇，类型为`ndarray`
    :return: DB指数
    '''
    # #********* Begin *********#
    label_set = np.unique(pred)
    mu = {}                 # 用于存放每个簇的中心点,key:簇的标签，value:中心点
    label_count = {}        # 用于存放每个簇的样本数,key:簇的标签，value:样本数
    # 计算簇的中点
    for label in label_set:
        mu[label] = np.zeros([len(feature[0])])  # 初始化为0，用于存放每个簇的中心点
        label_count[label] = 0             # 用于存放每个簇的样本数
    for i in range(len(pred)):
        mu[pred[i]] += feature[i]   # 求和,用于计算中心点
        label_count[pred[i]] += 1   # 计算每个簇的样本数
    for key in mu.keys():           # 计算中心点
        mu[key] /= label_count[key]  # 求平均值

    # 算数据到中心点的平均距离
    avg_d = {}          # 用于存放每个簇的样本到中心点的平均距离,key:簇的标签，value:样本到中心点的平均距离
    for label in label_set:
        avg_d[label] = 0        # 用于存放每个簇的样本到中心点的平均距离
    for i in range(len(pred)):
        avg_d[pred[i]] += np.sqrt(np.sum(np.square(feature[i] - mu[pred[i]])))   # 计算每个簇的样本到中心点的距离之和
    for key in mu.keys():
        avg_d[key] /= label_count[key]   # 计算每个簇的样本到中心点的平均距离


    # 算两个簇的中点之间的距离
    cen_d = []
    for i in range(len(label_set)):
        for j in range(len(label_set)):
            if i != j:
                t = {'c1': label_set[i], 'c2': label_set[j],
                     'dist': np.sqrt(np.sum(np.square(mu[label_set[i]] - mu[label_set[j]])))}  # 计算两个簇的中心点之间的距离，c1,c2为簇的标签，dist为其中心点之间的距离
                cen_d.append(t)
    dbi = 0
    for i in label_set:
        max_item = 0   # 用于存放每个簇的最大值
        for j in label_set:
            if i != j:  # 两个簇不相等
                for p in range(len(cen_d)):     # 对于每一对簇，计算两个簇的中心点之间的距离
                    if i == cen_d[p]['c1'] and j == cen_d[p]['c2']:  # 找到两个簇的中心点之间的距离
                        tmp = (avg_d[i] + avg_d[j]) / cen_d[p]['dist']
                        if tmp > max_item:
                            max_item = tmp
        dbi += max_item   # 求和
    dbi /= len(label_set)   # 求平均值
    return dbi
    # #********* End *********#


def calc_DI(feature, pred):
    '''
    计算并返回Dunn指数
    :param feature: 待聚类数据的特征，类型为`ndarray`
    :param pred: 聚类后数据所对应的簇，类型为`ndarray`
    :return: Dunn指数
    '''
    #********* Begin *********#
    label_set = np.unique(pred)     # 获取簇的标签
    min_d = []      # 用于存放两个簇之间的最短距离
    for i in range(len(label_set)):
        for j in range(i+1, len(label_set)):
            t = {'c1': label_set[i], 'c2': label_set[j], 'dist': np.inf}    # c1,c2为簇的标签，dist为两个簇之间的最短距离
            min_d.append(t)

    # 计算两个簇之间的最短距离
    for i in range(len(feature)):
        for j in range(len(feature)):
            for p in range(len(min_d)):
                if min_d[p]['c1'] == pred[i] and min_d[p]['c2'] == pred[j]: # 找到两个簇之间的样本
                    d = np.sqrt(np.sum(np.square(feature[i] - feature[j]))) # 计算两个簇之间的距离
                    if d < min_d[p]['dist']:    # 更新两个簇之间的最短距离
                        min_d[p]['dist'] = d    

    # 计算同一个簇中距离最远的样本对的距离
    max_diam = 0    # 用于存放同一个簇中距离最远的样本对的距离
    for i in range(len(feature)):
        for j in range(len(feature)):
            if pred[i] == pred[j]:      # 同一个簇中的样本
                d = np.sqrt(np.sum(np.square(feature[i] - feature[j])))
                if d > max_diam:
                    max_diam = d

    di = np.inf     # 用于存放Dunn指数
    for i in range(len(label_set)):     # 对于每一个簇
        for j in range(len(label_set)):
            for p in range(len(min_d)):
                d = min_d[p]['dist'] / max_diam
                if d < di:
                    di = d
    return di
    #********* End *********#
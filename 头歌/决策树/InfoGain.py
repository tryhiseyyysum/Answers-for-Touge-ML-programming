import numpy as np


def calcInfoGain(feature, label, index):
    '''
    计算信息增益
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:信息增益，类型float
    '''

    #*********** Begin ***********#
    # 计算信息熵
    def calcEntropy(label):
        entropy = 0
        label_set = set(label)
        for l in label_set:
            p = np.sum(label == l) / len(label)
            entropy -= p * np.log2(p)
        return entropy
    
    # 计算条件熵
    def calcCondEntropy(feature, label, index):
        cond_entropy = 0
        feature_set = set(feature[:, index])
        for f in feature_set:
            mask = feature[:, index] == f
            cond_entropy += np.sum(mask) / len(label) * calcEntropy(label[mask])
        return cond_entropy
    
    # 计算信息增益
    info_gain = calcEntropy(label) - calcCondEntropy(feature, label, index)
    return info_gain
    #*********** End *************#

feature=np.array([[0, 1], [1, 0], [1, 2], [0, 0], [1, 1]])
label=np.array([0, 1, 0, 0, 1])
print(calcInfoGain(feature, label, 0))
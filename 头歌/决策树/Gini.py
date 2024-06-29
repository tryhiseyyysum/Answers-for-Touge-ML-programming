import numpy as np

def calcGini(feature, label, index):
    '''
    计算基尼系数
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:基尼系数，类型float
    '''

    #********* Begin *********#
    def calGini(label):
        label_set=set(label)
        n=len(label)
        sum_p=0
        for l in label_set:
            p=np.sum(label==l)/n
            sum_p+=p**2
        return 1-sum_p
    
    feature_set=set(feature[:,index])  #当前属性的可取值
    n=len(label)
    res=0
    for f in feature_set:
        D_v_D=np.sum(feature[:,index]==f)/n  #D_v/D
        mask = feature[:,index]==f  #若当前属性取值为f，则mask对应的值为True，去label列表里面就能把它们对应的标签取出来放到Gini的计算函数中求
        res+=D_v_D*calGini(label[mask])
    return res
    #********* End *********#
import numpy as np

def TP(y_true, y_predict):
    return np.sum((y_true == 1) & (y_predict == 1))
    
def TN(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 0))
    
def FP(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 1))
    
def FN(y_true, y_predict):
    return np.sum((y_true == 1) & (y_predict == 0))

def precision_score(y_true, y_predict):
    '''
    计算精准率并返回
    :param y_true: 真实类别，类型为ndarray
    :param y_predict: 预测类别，类型为ndarray
    :return: 精准率，类型为float
    '''

    #********* Begin *********#
    return TP(y_true, y_predict)/(TP(y_true, y_predict)+FP(y_true, y_predict))
    #********* End *********#


def recall_score(y_true, y_predict):
    '''
    计算召回率并召回
    :param y_true: 真实类别，类型为ndarray
    :param y_predict: 预测类别，类型为ndarray
    :return: 召回率，类型为float
    '''

    #********* Begin *********#
    return TP(y_true, y_predict)/(TP(y_true, y_predict)+FN(y_true, y_predict))
    #********* End *********#

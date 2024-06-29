import numpy as np

def calc_JC(y_true, y_pred):
    '''
    计算并返回JC系数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: JC系数
    '''

    #******** Begin *******#
    m=y_true.shape[0]    #样本个数
    a,b,c,d=0.0,0.0,0.0,0.0
    for i in range(m):
        for j in range(i+1,m):
            if y_pred[i]==y_pred[j] and y_true[i]==y_true[j]:
                a+=1
            elif y_pred[i]==y_pred[j] and y_true[i]!=y_true[j]:
                b+=1
            elif y_pred[i]!=y_pred[j] and y_true[i]==y_true[j]:
                c+=1
            elif y_pred[i]!=y_pred[j] and y_true[i]!=y_true[j]:
                d+=1    
    JC=a/(a+b+c)   
    return JC              
    #******** End *******#


def calc_FM(y_true, y_pred):
    '''
    计算并返回FM指数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: FM指数
    '''

    #******** Begin *******#
    m=y_true.shape[0]    #样本个数
    a,b,c,d=0.0,0.0,0.0,0.0
    for i in range(m):
        for j in range(i+1,m):
            if y_pred[i]==y_pred[j] and y_true[i]==y_true[j]:
                a+=1
            elif y_pred[i]==y_pred[j] and y_true[i]!=y_true[j]:
                b+=1
            elif y_pred[i]!=y_pred[j] and y_true[i]==y_true[j]:
                c+=1
            elif y_pred[i]!=y_pred[j] and y_true[i]!=y_true[j]:
                d+=1    
    FM=np.sqrt((a/(a+b))*(a/(a+c)))
    return FM   
    #******** End *******#

def calc_Rand(y_true, y_pred):
    '''
    计算并返回Rand指数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: Rand指数
    '''

    #******** Begin *******#
    m=y_true.shape[0]    #样本个数
    a,b,c,d=0.0,0.0,0.0,0.0
    for i in range(m):
        for j in range(i+1,m):
            if y_pred[i]==y_pred[j] and y_true[i]==y_true[j]:
                a+=1
            elif y_pred[i]==y_pred[j] and y_true[i]!=y_true[j]:
                b+=1
            elif y_pred[i]!=y_pred[j] and y_true[i]==y_true[j]:
                c+=1
            elif y_pred[i]!=y_pred[j] and y_true[i]!=y_true[j]:
                d+=1    
    Rand=2*(a+d)/(m*(m-1))
    return Rand  
    #******** End *******#
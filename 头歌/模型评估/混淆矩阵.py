import numpy as np

def confusion_matrix(y_true, y_predict):
    '''
    构建二分类的混淆矩阵，并将其返回
    :param y_true: 真实类别，类型为ndarray
    :param y_predict: 预测类别，类型为ndarray
    :return: shape为(2, 2)的ndarray
    '''

    #********* Begin *********#
    def TP(y_true, y_predict):
        return np.sum((y_true==1) & (y_predict==1))
    
    def TN(y_true, y_predict):
        return np.sum((y_true==0) & (y_predict==0))
    
    def FP(y_true, y_predict):
        return np.sum((y_true==0) & (y_predict==1))
    
    def FN(y_true, y_predict):
        return np.sum((y_true==1) & (y_predict==0))
    
    confusion=np.zeros((2,2),dtype=int)
    confusion[0,0]=TN(y_true, y_predict)
    confusion[0,1]=FP(y_true, y_predict)
    confusion[1,0]=FN(y_true, y_predict)
    confusion[1,1]=TP(y_true, y_predict)
    
    return confusion
    #********* End *********#

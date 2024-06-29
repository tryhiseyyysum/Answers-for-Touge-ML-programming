#encoding=utf8 
import numpy as np
#mse
def mse_score(y_predict,y_test):
    mse = np.mean((y_predict-y_test)**2)
    return mse
#r2
def r2_score(y_predict,y_test):
    '''
    input:y_predict(ndarray):预测值
          y_test(ndarray):真实值
    output:r2(float):r2值
    '''
    #********* Begin *********#
    y_mean=np.mean(y_test)
    r2=1-np.sum((y_predict-y_test)**2)/(np.sum((y_mean-y_test)**2))
    #********* End *********#
    return r2
class LinearRegression :
    def __init__(self):
        '''初始化线性回归模型'''
        self.theta = None
    def fit_normal(self,train_data,train_label):
        '''
        input:train_data(ndarray):训练样本
              train_label(ndarray):训练标签
        '''
        #********* Begin *********#
        b=np.ones([train_data.shape[0],1])
        x=np.hstack((b,train_data))
        y=train_label
        self.theta=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        #********* End *********#
        return self
    def predict(self,test_data):
        '''
        input:test_data(ndarray):测试样本
        '''
        #********* Begin *********#
        b=np.ones([test_data.shape[0],1])
        x=np.hstack((b,test_data))
        return x.dot(self.theta)
        #********* End *********#

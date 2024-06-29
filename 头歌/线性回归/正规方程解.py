#encoding=utf8 
import numpy as np
def mse_score(y_predict,y_test):
    '''
    input:y_predict(ndarray):预测值
          y_test(ndarray):真实值
    ouput:mse(float):mse损失函数值
    '''
    #********* Begin *********#
    n=y_test.shape[0]
    mse=np.mean((y_test-y_predict)**2);
    #********* End *********#
    return mse
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
        x = np.c_[np.ones((train_data.shape[0], 1)), train_data]
        
        y=train_label
        self.theta=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        #********* End *********#
        return self.theta
    def predict(self,test_data):
        '''
        input:test_data(ndarray):测试样本
        '''
        #********* Begin *********#
        x = np.c_[np.ones((test_data.shape[0], 1)), test_data]
     
        return x.dot(self.theta)
        #********* End *********#
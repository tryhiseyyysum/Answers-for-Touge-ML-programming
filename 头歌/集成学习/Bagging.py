
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class BaggingClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        #分类器的数量，默认为10
        self.n_model = n_model
        #用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []


    def fit(self, feature, label):
        '''
        训练模型，请记得将模型保存至self.models
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''

        #************* Begin ************#
        #feature.shape=(135,4)
        for i in range(self.n_model):
            #随机选择数据
            index = np.random.choice(range(len(feature)),len(feature),replace=True) 
            #index.shape(135,)
            #构建决策树
            clf = DecisionTreeClassifier()
            clf.fit(feature[index],label[index])
            self.models.append(clf)
        #************* End **************#


    def predict(self, feature):
        '''
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        #************* Begin ************#
        #feature.shape=(15,4)，15个样本，每个样本4个特征
        predictions=np.array([model.predict(feature) for model in self.models])
        #predictions.shape(10,15)  10个分类器，15个样本，每一行是一个分类器对于所有15个测试样本的分类结果
        res=np.array(
            [np.bincount(sample).argmax() for sample in np.array(predictions.T)]
        )   #现在要转置使得每行表示一个样本，遍历每个样本，求出所有10个分类器对其类别投票的最高票数的那个类别，就是这个样本最终的预测类别了

        return res
        #************* End **************#

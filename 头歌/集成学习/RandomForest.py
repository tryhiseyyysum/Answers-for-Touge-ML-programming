
import numpy as np

#建议代码，也算是Begin-End中的一部分
from collections import  Counter
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        #分类器的数量，默认为10
        self.n_model = n_model
        #用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []
        #用于保存决策树训练时随机选取的列的索引
        self.col_indexs = []


    def fit(self, feature, label):
        '''
        训练模型
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''

        #************* Begin ************#
        for i in range(self.n_model):
            #随机选择数据
            #feature.shape=(512,30)
            data_indices = np.random.choice(range(len(feature)),len(feature),replace=True) 
            #data_indices.shape(512,)
            #构建决策树
            bootstrap_feature = feature[data_indices]
            bootstrap_label = label[data_indices]

            #bootstrap_feature.shape=(512,30)
            # 随机属性选择
            attr_indices = np.random.choice(
                range(len(bootstrap_feature[0])),
                int(np.log(len(bootstrap_feature[0]))),
                replace=False
            )
            #len(bootstrap_feature[0])=30，代表决策树可划分属性的个数，不放回地选择log(k)个属性进行划分，replace=False
            #attr_indices.shape=(3,)
            self.col_indexs.append(attr_indices) #将随机选择的划分属性存入列表中
            bootstrap_sub_feature = bootstrap_feature[:, attr_indices]  #只挑选出刚刚随机选出的需要进行划分的属性的那几列的样本，用于训练决策树基学习器

            clf = DecisionTreeClassifier()
            clf.fit(bootstrap_sub_feature, bootstrap_label)
            self.models.append(clf)
        #************* End **************#


    def predict(self, feature):
        '''
        :param feature:测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        #************* Begin ************#
        predictions = np.array(
            [
                clf.predict(feature[:, self.col_indexs[i]]) #在预测时所用到的特征必须与训练模型时所用到的特征保持一致
                for i, clf in enumerate(self.models)
            ]
        ).T

        return np.array(
            [
                np.bincount(sample_predictions).argmax()
                for sample_predictions in predictions
            ]
        )
        #************* End **************#

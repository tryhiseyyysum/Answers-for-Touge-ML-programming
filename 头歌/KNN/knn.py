#encoding=utf8
import numpy as np

class kNNClassifier(object):
    def __init__(self, k):
        '''
        初始化函数
        :param k:kNN算法中的k
        '''
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.train_feature = None
        # 用来存放训练标签，类型为ndarray
        self.train_label = None


    def fit(self, feature, label):
        '''
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        '''

        #********* Begin *********#
        self.train_feature=feature
        self.train_label=label
        #********* End *********#


    def predict(self, feature):
        '''
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        '''

        #********* Begin *********#
        result=[]
        for i in range(len(feature)):
            #feature.shape=(38,4)  38个测试样本，特征有4个
            #计算测试数据与训练数据的欧式距离
            distance=np.sum((self.train_feature-feature[i])**2,axis=1)**0.5
            #self.train_feature.shape=(112,4) 训练集有112个样本，每个样本特征有4个
            #distance.shape=(112,)，表示当前测试样本i和112个训练样本的距离
            #对距离排序，取出前k个
            index=np.argsort(distance)[:self.k]
            #取出前k个标签
            label=self.train_label[index]
            #找到k个标签中出现次数最多的标签
            result.append(np.argmax(np.bincount(label)))
        return np.array(result)
        
        #********* End *********#

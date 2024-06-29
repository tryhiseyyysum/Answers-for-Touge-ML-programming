import numpy as np

#本代码完全自己写出来的

class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率，即先验概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率，即似然(给定类别的条件概率)
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.condition_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}
    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''


        #********* Begin *********#

        #计算先验概率
        for l in np.unique(label):
            self.label_prob[l]=np.sum(label==l)/len(label)  #求该类别的先验概率
            self.condition_prob[l]={}   #创建空字典

        #计算类条件概率
        for l in np.unique(label):
            for index,f in enumerate(feature.T): #feature.T之后每一行是一个特征，f如[1,2,3,4,5]，表示所有样本的某一个特征取值
                self.condition_prob[l][index]={}  #创建空字典
                for ff in np.unique(f):  #对于每一个特征的取值，统计该类别下该特征取值的概率
                    self.condition_prob[l][index][ff]=np.sum((label==l) & (f==ff))/np.sum(label==l)
        #*******    ** End *********#


    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        # ********* Begin *********#
        res=[]
        for f in feature:  #对于每一个样本
            predict_label=None  #维护一个最大概率的类别
            max_prob=-1
            for k,l in self.label_prob.items(): #k是类别，l是先验概率
                conditon_prob=1
                for index,ff in enumerate(f):
                    conditon_prob*=self.condition_prob[k][index][ff]  #计算条件概率，连乘
                prior_prob=l  #该类的先验概率值
                prob=conditon_prob*prior_prob  #先验概率*条件概率
                if prob>max_prob:  #维护所有类别中最大的概率，以及对应的类别，这个最大的概率的类别就是预测的类别
                    max_prob=prob
                    predict_label=k
            res.append(predict_label)  #将当前这个样本的预测类别加入到结果中

        return np.array(res)

        #********* End *********#
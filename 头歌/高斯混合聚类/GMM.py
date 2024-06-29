import numpy as np
from scipy.stats import multivariate_normal

class GMM(object):
    def __init__(self, n_components, max_iter=100):
        '''
        构造函数
        :param n_components: 想要划分成几个簇，类型为int
        :param max_iter: EM的最大迭代次数
        '''
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, train_data):
        '''
        训练，将模型参数分别保存至self.alpha，self.mu，self.sigma中
        :param train_data: 训练数据集，类型为ndarray
        :return: 无返回
        '''
        row, col = train_data.shape
        # 初始化每个高斯分布的响应系数
        self.alpha = np.array([1.0 / self.n_components] * self.n_components)
        # 初始化每个高斯分布的均值向量
        self.mu = np.random.rand(self.n_components, col)
        # 初始化每个高斯分布的协方差矩阵
        self.sigma = np.array([np.eye(col)] * self.n_components)

        #********* Begin *******#
        for _ in range(self.max_iter):
            # E step
            gamma = np.zeros((row, self.n_components))   #后验概率

            # ---------------根据贝叶斯公式计算后验概率的过程---------------
            for i in range(self.n_components):      #对于一个聚类簇i，计算每个样本属于这个簇的概率
                gamma[:, i] = self.alpha[i] * multivariate_normal.pdf(train_data, self.mu[i], self.sigma[i])  #西瓜书公式9.30
            gamma /= gamma.sum(axis=1).reshape(-1, 1)   #归一化
            #-----------------------------------------------------------

            # M step   (更新参数，要用到E步中计算出来的后验概率gamma)
            N = gamma.sum(axis=0)      # 所有样本的后验概率求和，得到的是每个簇的样本后验概率之和
            for i in range(self.n_components):
                self.mu[i] = np.dot(gamma[:, i], train_data) / N[i]  #西瓜书公式9.34
                self.sigma[i] = np.dot((train_data - self.mu[i]).T, gamma[:, i].reshape(-1, 1) * (train_data - self.mu[i])) / N[i]  #西瓜书公式9.35
                # N[i]为簇i的样本后验概率之和，即簇i的样本数
                self.alpha[i] = N[i] / row  #西瓜书公式9.38

        #********* End *********#


    def predict(self, test_data):
        '''
        预测，根据训练好的模型参数将test_data进行划分。
        注意：划分的标签的取值范围为[0,self.n_components-1]，即若self.n_components为3，则划分的标签的可能取值为0,1,2。
        :param test_data: 测试集数据，类型为ndarray
        :return: 划分结果，类型为你ndarray
        '''

        #********* Begin *********#
        row = test_data.shape[0]   #样本个数
        gamma = np.zeros((row, self.n_components))
        for i in range(self.n_components):      #对于每一个聚类簇i，计算每个样本属于这个簇的概率
            #计算后验概率，因为是比较大小，所以不用除以分母了，分母大家都一样
            gamma[:, i] = self.alpha[i] * multivariate_normal.pdf(test_data, self.mu[i], self.sigma[i])  #西瓜书公式9.30
        return gamma.argmax(axis=1)  #每个样本在k个簇里面的概率最大值对应的簇标签
    
        #********* End *********#

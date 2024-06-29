#encoding=utf8
import numpy as np
#adaboost算法
class AdaBoost:
    '''
    input:n_estimators(int):迭代轮数
          learning_rate(float):弱分类器权重缩减系数
    '''
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators  # 弱分类器数目
        self.learning_rate = learning_rate
    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape  # M为样本数，N为特征数
        # 弱分类器数目和集合
        self.clf_sets = []
        # 初始化weights
        self.weights = [1.0/self.M]*self.M  # 初始化样本权重系数，[1.0/self.M]*self.M表示生成一个长度为self.M的列表，每个元素都是1.0/self.M，即复制样本个数份
        # G(x)系数 alpha
        self.alpha = []
    #********* Begin *********#
    def _G(self, features, labels, weights):  # 弱分类器
        '''
        input:features(ndarray):数据特征
              labels(ndarray):数据标签
              weights(ndarray):样本权重系数
        '''
        m = len(features)
        error = 100000.0
        best_v = 0.0
        #单维features
        features_min = min(features)  
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate)//self.learning_rate #计算步长,即特征值的范围/学习率
        direct, compare_array = None, None  #初始化方向和分类结果
        for i in range(1, int(n_step)):     #对于每个特征值
            v = features_min + self.learning_rate * i  #计算阈值
            if v not in features:   #如果阈值不在特征值中
                #误分类计算
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])  #对于每个样本，如果特征大于阈值，标签为1，否则为-1
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])  #把所有分类错的权重都加起来
                compare_array_negative = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_negative = sum([weights[k] for k in range(m) if compare_array_negative[k] != labels[k]])
                if weight_error_positive < weight_error_negative:  #选择误差小的，正类别误差小，则模型输出结果为正类别，否则为负类别
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_negative
                    _compare_array = compare_array_negative
                    direct = 'negative'

                if weight_error < error: #选择误差最小的
                    error = weight_error #更新误差
                    compare_array = _compare_array  #更新分类结果
                    best_v = v  #更新阈值
        return best_v, direct, error, compare_array

    # 计算alpha
    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    # 规范化因子
    def _Z(self, weights, a, clf):
        #weights:样本权重系数,a:G(x)系数,即alpha,clf:弱分类器，即G(x)
        return sum([weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) for i in range(self.M)])  #对于每个样本，计算权值*exp(-1*alpha*y*g(x))
    # 权值更新
    def _w(self, a, clf, Z):
        for i in range(self.M):  #对于每个样本
            self.weights[i] = self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) / Z  #样本分布更新公式

    # G(x)的线性组合
    def G(self, x, v, direct):
        #x:样本特征，v:特征阈值，direct:正负类别
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1
        
    def fit(self, X, y):
        '''
        X(ndarray):训练数据
        y(ndarray):训练标签
        '''

        self.init_args(X, y)
        for epoch in range(self.clf_num): #对于每个弱分类器
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度，选择误差最小的
            for j in range(self.N):  #对于每个特征
                features = self.X[:, j]
                #分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)  #获得弱分类器
                if error < best_clf_error:
                    best_clf_error = error  #更新最小误差
                    best_v = v         #更新最佳阈值
                    final_direct = direct       #更新最佳方向
                    clf_result = compare_array      #更新最佳分类结果
                    axis = j        #更新最佳特征
                if best_clf_error == 0: #如果误差为0，即分类正确，跳出循环
                    break
            # 计算G(x)系数a
            a = self._alpha(best_clf_error)
            self.alpha.append(a)
        
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))  #记录弱分类器,包括特征，阈值，方向

            # 规范化因子
            Z = self._Z(self.weights, a, clf_result)

            # 权值更新
            self._w(a, clf_result, Z)

    def predict(self, data):
        '''
        input:data(ndarray):单个样本
        output:预测为正样本返回+1，负样本返回-1
        '''
        res = 0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]  #获取弱分类器,包括特征，阈值，方向
            f_input = data[axis]    #获取特征值
            res += self.alpha[i] * self.G(f_input, clf_v, direct)   #计算G(x)的线性组合
        return 1 if res > 0 else -1

    #********* End *********#

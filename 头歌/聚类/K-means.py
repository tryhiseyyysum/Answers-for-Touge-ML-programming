#encoding=utf8
import numpy as np

# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances

class Kmeans():
    """Kmeans聚类算法.
    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数. 
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 
        则说明算法已经收敛
    """
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        np.random.seed(1)
    #********* Begin *********#
    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))  #初始化聚类中心
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]  #随机选择一个样本作为聚类中心np.random.choice不写size参数默认返回1个随机数
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids) #计算该样本与所有聚类中心的距离
        closest_i = np.argmin(distances)  #返回最小值的索引
        return closest_i
    
    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = X.shape[0]
        clusters = [[] for _ in range(self.k)]  #初始化聚类结果
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)  #找到该样本最近的聚类中心
            clusters[centroid_i].append(sample_i)  #第i个聚类中心拥有的样本
        return clusters
    
    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):  #cluster是一个列表，里面存放的是属于第i个聚类的样本的索引
            centroid = np.mean(X[cluster], axis=0)  #计算第i个聚类的新的聚类中心
            centroids[i] = centroid     #更新第i个聚类的聚类中心
        return centroids
    

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(X.shape[0])
        for cluster_i, cluster in enumerate(clusters):  #cluster是一个列表，里面存放的是属于第i个聚类的样本的索引
            for sample_i in cluster:    #遍历第i个聚类中的所有样本
                y_pred[sample_i] = cluster_i    #将第i个聚类中的所有样本的类别标签都设置为i
        return y_pred
    
    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)
        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            # 计算新的聚类中心
            prev_centroids = centroids
            centroids = self.update_centroids(clusters, X)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - prev_centroids
            #只要有一个聚类中心的变化小于varepsilon，就认为算法已经收敛，退出迭代
            if diff.any() < self.varepsilon:  #.any()表示只要有一个元素满足条件就返回True
                break
        return self.get_cluster_labels(clusters, X)

    #********* End *********#

#encoding=utf8
import numpy as np
import random
#寻找样本点j的eps邻域内的点
def findNeighbor(j,X,eps):
    N=[]
    for p in range(X.shape[0]):   #找到所有领域内对象
        temp=np.sqrt(np.sum(np.square(X[j]-X[p])))   #欧氏距离
        if(temp<=eps):
            N.append(p)
    return N

#dbscan算法
def dbscan(X,eps,min_Pts):
    '''
    input:X(ndarray):样本数据
          eps(float):eps邻域半径
          min_Pts(int):eps邻域内最少点个数
    output:cluster(list):聚类结果
    '''
    #********* Begin *********#
    main_point=[]   #核心点集
    for sample_i in range(X.shape[0]):
        N=findNeighbor(sample_i,X,eps)
        if len(N)>=min_Pts:
            main_point.append(sample_i)

    k = 0  # 聚类簇数
    cluster = [-1] * X.shape[0]  # 聚类结果

    while len(main_point) > 0:
        i = random.choice(main_point)   #在核心点集中随机选取一个核心对象
        cluster[i] = k   #将核心对象标记为当前簇

        #找到核心点的所有密度可达点的关键在于使用队列queue，将邻域内的对象加入队列，使得下次取出队列中的对象时，可以继续找到邻居的邻居，即找到所有密度可达点
        #思想有点类似广度优先BFS
        queue = [i]
        while len(queue) > 0:
            q = queue.pop(0)    #取出队列中的第一个对象
            neighbor = findNeighbor(q, X, eps)  #找到q的eps邻域内的对象
            for p in neighbor:      #遍历q的邻域内对象
                if cluster[p] == -1:    #邻居是密度直达的，如果该邻居还没有被聚类，则加入当前聚类簇中
                    cluster[p] = k   #将该对象标记为当前簇
                    queue.append(p)     #将该邻居加入队列
            if len(neighbor) >= min_Pts:    #如果q是核心对象
                main_point.remove(q)    #将q从核心对象集中移除

        k += 1  # 聚类簇数加1
    
    #********* End *********#
    return cluster

#测试
data = np.array([[1,2],[2,2],[2,3],[8,7],[8,8],[25,80]])
eps=3
min_Pts=2
result=dbscan(data,eps,min_Pts)
print(result)
#输出：[0, 0, 0, 1, 1, -1]  #其中-1表示噪声点，最终聚类簇数为2
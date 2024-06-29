import numpy as np

def calAUC(prob, labels):
    '''
    计算AUC并返回
    :param prob: 模型预测样本为Positive的概率列表，类型为ndarray
    :param labels: 样本的真实类别列表，其中1表示Positive，0表示Negtive，类型为ndarray
    :return: AUC，类型为float
    '''

    #********* Begin *********#
    M=np.sum(labels)
    N=len(labels)-M

    sorted_indices=np.argsort(prob)
    sorted_labels=labels[sorted_indices]

    rank=0
    for i in range(len(sorted_labels)):
        if sorted_labels[i]==1:
            rank+=(i+1)
    
    return (rank-(M*(M+1)/2))/(M*N)
    #********* End *********#

#encoding=utf8
from sklearn.cluster import DBSCAN
def data_cluster(data):
    '''
    input: data(ndarray) :数据
    output: result(ndarray):聚类结果
    '''
    #********* Begin *********#
    db=DBSCAN(eps=0.5,min_samples=10)
    res=db.fit_predict(data)
    return res
    #********* End *********#                     
                                         
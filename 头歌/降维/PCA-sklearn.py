from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

def cancer_predict(train_sample, train_label, test_sample):
    '''
    使用PCA降维，并进行分类，最后将分类结果返回
    :param train_sample:训练样本, 类型为ndarray
    :param train_label:训练标签, 类型为ndarray
    :param test_sample:测试样本, 类型为ndarray
    :return: 分类结果
    '''

    #********* Begin *********#

    #降维
    pca=PCA(n_components=11)        #降维到11维
    pca_train_sample=pca.fit_transform(train_sample)
    pca_test_sample=pca.transform(test_sample)
    
    #分类，PCA只是把特征进行了降维，分类还是要用分类器
    svc=LinearSVC()
    svc.fit(pca_train_sample,train_label)
    res=svc.predict(pca_test_sample)
    return res
    #********* End *********#
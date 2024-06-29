from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


def news_predict(train_sample, train_label, test_sample):
    '''
    训练模型并进行预测，返回预测结果
    :param train_sample:原始训练集中的新闻文本，类型为ndarray
    :param train_label:训练集中新闻文本对应的主题标签，类型为ndarray
    :param test_sample:原始测试集中的新闻文本，类型为ndarray
    :return 预测结果，类型为ndarray
    '''

    #********* Begin *********#

    #1. 先把文本转换成词频矩阵，即向量化
    vc=CountVectorizer()
    train_sample_count_vectorizer=vc.fit_transform(train_sample)
    test_sample_count_vectorizer=vc.transform(test_sample)

    #2. 计算tf-idf，用来得到训练数据和测试数据
    tfidf=TfidfTransformer()
    X_train=tfidf.fit_transform(train_sample_count_vectorizer)
    X_test=tfidf.transform(test_sample_count_vectorizer)

    clf=MultinomialNB(alpha=0.01)   #alpha=0.01是拉普拉斯平滑系数
    clf.fit(X_train,train_label)
    res=clf.predict(X_test)
    return res
    #********* End *********#

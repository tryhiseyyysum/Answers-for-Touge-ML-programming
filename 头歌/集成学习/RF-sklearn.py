from sklearn.ensemble import RandomForestClassifier

def digit_predict(train_image, train_label, test_image):
    '''
    实现功能：训练模型并输出预测结果
    :param train_image: 包含多条训练样本的样本集，类型为ndarray,shape为[-1, 8, 8]
    :param train_label: 包含多条训练样本标签的标签集，类型为ndarray
    :param test_image: 包含多条测试样本的测试集，类型为ndarry
    :return: test_image对应的预测标签，类型为ndarray
    '''

    #************* Begin ************#
    X_train=train_image.reshape(-1,8*8)
    X_test=test_image.reshape(-1,8*8)
    rf=RandomForestClassifier(n_estimators=20)
    rf.fit(X_train,train_label)
    res=rf.predict(X_test)
    return res
    #************* End **************#
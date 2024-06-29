from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def digit_predict(train_image, train_label, test_image):
    '''
    实现功能：训练模型并输出预测结果
    :param train_sample: 包含多条训练样本的样本集，类型为ndarray,shape为[-1, 8, 8]
    :param train_label: 包含多条训练样本标签的标签集，类型为ndarray
    :param test_sample: 包含多条测试样本的测试集，类型为ndarry
    :return: test_sample对应的预测标签
    '''

    #************* Begin ************#
    train_data=train_image.reshape(-1,8*8)
    test_data=test_image.reshape(-1,8*8)
    #标准化数据
    scaler=StandardScaler()
    train_data=scaler.fit_transform(train_data)
    test_data=scaler.transform(test_data)

    lr=LogisticRegression(solver='lbfgs',max_iter=10,C=10)
    lr.fit(train_data,train_label)
    res=lr.predict(test_data)
    return res
    #************* End **************#
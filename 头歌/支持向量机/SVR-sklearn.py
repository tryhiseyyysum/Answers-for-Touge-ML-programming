#encoding=utf8
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def svr_predict(train_data, train_label, test_data):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
          test_data(ndarray):测试数据
    output:predict(ndarray):测试集预测标签
    '''
    #********* Begin *********#
    
    # 标准化数据
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    
    # 创建SVR模型
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    # 训练模型
    model.fit(train_data, train_label)
    
    # 进行预测
    predict = model.predict(test_data)
    
    #********* End *********#
    return predict

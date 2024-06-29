#encoding=utf8
from sklearn.svm import LinearSVC


def linearsvc_predict(train_data,train_label,test_data):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
    output:predict(ndarray):测试集预测标签
    '''
    #********* Begin *********# 
    svc=LinearSVC(penalty="l2", 
        loss="squared_hinge",
        dual=False,
        tol=0.001,
        C=0.5,
        multi_class="ovr",
        fit_intercept=False,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000)
    svc.fit(train_data,train_label)
    predict=svc.predict(test_data)
    #********* End *********# 
    return predict
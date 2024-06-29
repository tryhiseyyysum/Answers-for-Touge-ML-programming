from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def classification_performance(y_true, y_pred, y_prob):
    '''
    返回准确度、精准率、召回率、f1 Score和AUC
    :param y_true:样本的真实类别，类型为`ndarray`
    :param y_pred:模型预测出的类别，类型为`ndarray`
    :param y_prob:模型预测样本为`Positive`的概率，类型为`ndarray`
    :return:
    '''

    #********* Begin *********#
    acc=accuracy_score(y_true, y_pred)
    P=precision_score(y_true, y_pred)
    R=recall_score(y_true, y_pred)
    f1=f1_score(y_true, y_pred)
    roc=roc_auc_score(y_true, y_prob)
    return acc,P,R,f1,roc
    #********* End *********#
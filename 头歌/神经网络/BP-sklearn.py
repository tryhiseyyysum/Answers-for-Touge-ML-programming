#encoding=utf8
import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

X_train=pd.read_csv('./step2/train_data.csv')
Y_train=pd.read_csv('./step2/train_label.csv')
Y_train=Y_train['target']
X_test=pd.read_csv('./step2/test_data.csv')

# 输入层：根据 X_train 的特征数量确定
# 隐藏层1：10 个神经元
# 隐藏层2：5 个神经元
# 输出层：根据 Y_train 的类别数量确定

#输入层和输出层的神经元数量是固定的，不需要自己设置，隐藏层的神经元数量可以自己设置
mlp=MLPClassifier(solver='lbfgs',max_iter=100,alpha=1e-4,hidden_layer_sizes=(10,5)) #两个隐藏层，第一个隐藏层10个神经元，第二个隐藏层5个神经元
mlp.fit(X_train,Y_train)
res=mlp.predict(X_test)
df=pd.DataFrame(res,columns=['result'])
df.to_csv('./step2/result.csv',index=False)
#********* Begin *********#

#********* End *********#
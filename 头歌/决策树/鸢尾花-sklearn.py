#********* Begin *********#
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

#.as_matrix()可以将DataFrame转换成ndarray类型,这样才能正常的使用fit和predict
X_train=pd.read_csv('./step7/train_data.csv').as_matrix()  
Y_train=pd.read_csv('./step7/train_label.csv').as_matrix()
X_test=pd.read_csv('./step7/test_data.csv').as_matrix()

clf=DecisionTreeClassifier()
clf.fit(X_train,Y_train)
res=clf.predict(X_test)

df=pd.DataFrame(res,columns=['target'])
df.to_csv('./step7/predict.csv',index=False)
#********* End *********#
#encoding=utf8
#********* Begin *********#
import pandas as pd
from sklearn.linear_model import LinearRegression
train_data=pd.read_csv('./step3/train_data.csv')
train_label=pd.read_csv('./step3/train_label.csv')
train_label=train_label['target']
test_data=pd.read_csv('./step3/test_data.csv')

lr=LinearRegression()
lr.fit(train_data,train_label)
predict=lr.predict(test_data)
df=pd.DataFrame(predict,columns=['result'])
df.to_csv('./step3/result.csv')
#********* End *********#



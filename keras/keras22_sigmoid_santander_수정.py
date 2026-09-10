# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# path = './_data/kaggle_santander/'
path = 'c:/study/_data/kaggle_santander/'

train_csv = pd.read_csv(path + "train.csv",index_col=0)
test_csv = pd.read_csv(path +'test.csv',index_col=0)
submission_csv = pd.read_csv(path +'sample_submission.csv',index_col=0)

print(train_csv.shape)          #(200000, 201)
print(test_csv.shape)           #(200000, 201)
print(submission_csv.shape)     #(200000, 1)

# exit()
print(train_csv.info())
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

x = train_csv.drop(['target'],axis=1)
y = train_csv['target']
print(x.shape,y.shape)         #(200000, 200) (200000,)

print(np.unique(y,return_counts=True))
#(array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train , y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state=333,
    stratify=y,
)

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=200,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

start_time = time.time()
hist = model.fit(x_train,y_train,
                epochs=5,
                batch_size = 500,
                verbose = 1,
                validation_split=0.3
                 )
end_time = time.time()

#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print("========================================")
print('loss : ', loss[0])
print('acc : ', round(loss[1],4))
print("========================================")

y_pred = model.predict(test_csv)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test,y_pred)
print("acc_score : ", acc_score)

##############submission.csv 만들기 // count컬럼에 값 넣어준다################
print(submission_csv)
y_pred = np.round(model.predict(test_csv))
submission_csv['target']  = y_pred   

# print(submission_csv)
# print(submission_csv.shape)

submission_csv.to_csv(path + "submit/" + "submit_0908_1739.csv",)

'''
# print(pd.value_counts(y,sort=True))
train_test_split / stratify 사용
2.Modeling

input 200개
마지막 activation = sigmoid
batch size 크게 → 속도 빠르게
데이터 20만 개
시간상 train_size 줄이기
3. binary_crossentropy / Adam
compile에서 metrics = accuracy 넣기
4.  model.evaluate(x_test, y_test)
loss 출력
sklearn의 accuracy_score
sigmoid → 0~1 사이로 predict
np.round() → 반올림
→ submission 파일까지 만들어서 제출
'''
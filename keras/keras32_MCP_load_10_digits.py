from sklearn.datasets import load_digits
#[실습]acc 결과 1.0 목표 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model #🩷
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets= load_digits()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape , y.shape)    #(1797, 64) (1797,)
print(y)
print(np.unique(y, return_counts=True))
#array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

x_train , x_test , y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True,
    stratify=y,
)

print(x_train.shape, x_test.shape)  #(1437, 64) (360, 64)
print(y_train.shape, y_test.shape)  #(1437, 10) (360, 10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용

#2. 모델 구성 
path = './_save/keras31/'#🩷🩷🩷
model = load_model(path + '파일명.keras')#🩷🩷🩷

#3. 컴파일 훈련
model.save(path + '파일명.keras') #🩷🩷🩷


print("=======================================")

#4. 평가, 예측 
result = model.evaluate(x_test,y_test,)
print('loss: ',result[0])
print('acc: ',round(result[1],2))

y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy_score)
print(" 걸린시간 : ", round(end_time - start_time,2),"초")

'''


'''
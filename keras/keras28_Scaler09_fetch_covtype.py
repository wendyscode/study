from sklearn.datasets import fetch_covtype
# [실습] acc기준 0.93 이상 
#다중분류 / 데이터 불러와서  xy shape 확인 .다중분류 확인 / 시간 재고 배치크게!

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = fetch_covtype()
# print(datasets) #target = y
# print(datasets.DESCR) # 묘사하다
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)
print(y)
print(np.unique(y,return_counts=True))
#array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

from tensorflow.keras.utils import to_categorical
y_oh = to_categorical(y)
print(y_oh[:10])
print(y[:10])
print(y_oh.shape)  #(581012, 8)

# exit()
x_train , x_test, y_train , y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state= 333,  
    shuffle= True,
    stratify=y,  
)
print(x_train.shape,x_test.shape)   #(464809, 54) (116203, 54)
print(x_train.shape,y_test.shape)   #(464809, 54) (116203,)

y_train = pd.get_dummies(y_train, dtype=int).values
y_test = pd.get_dummies(y_test, dtype=int).values

print(y_train.shape)  # (464809, 7)
print(y_test.shape)   # (116203, 7)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용

#2. 모델구성 
model = Sequential()
model.add(Dense(10, input_dim =54, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련  
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience= 10,
    restore_best_weights= True,
)
start_time = time.time()
model.fit(x_train,y_train, epochs=10, batch_size=256,
          verbose=1,
          validation_split = 0.2,
          callbacks =[es]
          )
end_time = time.time()

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
성능비교 
기존 : 
acc_score :  0.7109110780272454
 걸린시간 :  17.39 초

Minmax : 
acc_score :  0.7447139918934967
 걸린시간 :  17.02 초
 
'''
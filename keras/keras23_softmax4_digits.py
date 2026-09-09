from sklearn.datasets import load_digits
#[실습]acc 결과 1.0 목표 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
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

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim = 64, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train,y_train, epochs=1000, batch_size=8,
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
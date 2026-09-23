from sklearn.datasets import load_wine
#[실습]acc = 0.95 이상이면 합격 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model #🩷
from tensorflow.keras.layers import Dense, Dropout #💛
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_wine()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(178, 13) (178,)
print(y)
print(np.unique(y, return_counts=True))
#(array([0, 1, 2]), array([59, 71, 48]))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

x_train , x_test , y_train , y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=4333,
    shuffle= True,
    stratify= y,
)
x_train, x_val , y_train, y_val = train_test_split(
                                    x_train, y_train,
                                   train_size = 0.5,
                                   random_state = 123,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = StandardScaler()
scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용

print(x_train.shape, x_test.shape)  #(142, 13) (36, 13)
print(y_train.shape, y_test.shape)  #(142, 3) (36, 3)

# exit()
# 2.모델구성 
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #🤎🤎🤎🤎🤎

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001 #디폴트
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=40,
        verbose=1,
        restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode= 'auto',
    patience=20,
    verbose=1,
    factor=0.5,

)

start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train,
                callbacks=[es, rlr ], verbose=1,
                  epochs=100, batch_size=32, validation_data=(x_val, y_val))
end_time = time.time() # 현재시간을 반환. 끝시간


print("=======================================")

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc :', round(result[1],2))

y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy_score)
print("걸린시간 : ", round(end_time - start_time,2),"초")


'''
acc_score :  0.9444444444444444
걸린시간 :  86.51 초

acc_score :  0.9722222222222222
걸린시간 :  2.13 초


'''
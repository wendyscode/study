from sklearn.datasets import load_wine
#[실습]acc = 0.95 이상이면 합격 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model #🩷
from tensorflow.keras.layers import Dense
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

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam',metrics = ['acc'],)

es = EarlyStopping(monitor='val_loss', mode= 'min',
                    patience= 30,
                    restore_best_weights= True,
                    verbose=1,
                   )
################## mcp 세이브 파일명 만들기 시작 #####################🩷🩷🩷🩷🩷
import datetime
date = datetime.datetime.now()  #현재시간반환
print(date)         #2026-09-14 11:41:15.177740
print(type(date))   #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)         #0914_1147
print(type(date))   #<class 'str'> : 문자형태

path = './_save/keras31/'
filename ='{epoch:04d}-{val_loss:.4f}.keras'
filepath ="".join([path,"k31_wine_",date,"-", filename])

# 내가 생각하는 파일명 예.
# './_save/keras30/' + "k30_" + "0914_1147" + '530-0.001.keras'
#################### mcp 세이브 파일명 만들기 끝 #####################🩷🩷🩷🩷🩷
# exit()

mcp = ModelCheckpoint(                      
    monitor='val_loss', 
    mode='auto', 
    save_best_only= True, 
    filepath=  filepath,   
    verbose=1,
)                                           

start_time = time.time()
hist = model.fit(x_train,y_train,
                 epochs=1000,batch_size=32,validation_split = 0.2,
                callbacks=[es,mcp],          
                verbose=1,
                 )
end_time = time.time()


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

'''
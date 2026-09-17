 #1.mnist 2.cifar10 3.cifar100
#41-1 카피 

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.metrics import accuracy_score

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape , y_train.shape)
print(x_test.shape, y_test.shape)

print(np.max(x_train),np.min(x_train))
print(np.max(x_test), np.min(x_test))

##### 스케일링 1 >>>>>>>>>>>0~255 → 0~1 변경 ###########
x_train = x_train/255.  #점찍으면 플로트(실수)표현
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) 
print(np.max(x_test), np.min(x_test))  

x_train = x_train.reshape(-1, 28 * 28)#🍧🍧🍧🍧🍧🍧
x_test = x_test.reshape(-1, 28 * 28)    #🍧🍧🍧🍧🍧🍧
print(x_train.shape, x_test.shape)  #(60000, 784) (10000, 784)
# exit()

#실습시작!! 
# 목표는 CNN을 이겨라!!! 

#2. 모델구성
model = Sequential()
model.add(Dense(128,input_shape = (28*28,),activation='relu'))#🍧🍧🍧🍧🍧🍧
model.add(Dense(64,activation='relu'))#🍧
model.add(Dense(32,activation='relu'))#🍧
model.add(Dense(10,activation='softmax'))#🍧

#3. 컴파일 , 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam',
            metrics = ['acc'],
            )   

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    patience= 30,
    restore_best_weights= True,
)

start_time = time.time()
model.fit(x_train,y_train, epochs=5000, batch_size=128,
        verbose = 1, 
        validation_split = 0.2,
         callbacks=[es],

          )
end_time = time.time()

#4. 평가예측
print("==============model.evaluate======================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('loss :', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict,axis=1,)#.reshape(-1,1)
# y_test = np.argmax(y_test,axis=1,)#.reshape(-1,1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')

'''
CNN
accuracy_score :  0.7644
걸린시간 :  393.2 초

DNN
accuracy_score :  0.9734
걸린시간 :  27.26 초

accuracy_score :  0.9756
걸린시간 :  28.17 초
'''
#36-2 카피

import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense,Dropout,Flatten

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)
print(np.max(x_train), np.min(x_train)) #255 0 #제일 밝은 값과 제일 어두운 값
print(np.max(x_test), np.min(x_test))   #255 0  #0=검정 255=흰색

##### 스케일링 1 >>>>>>>>>>>0~255 → 0~1 변경 ###########
x_train = x_train/255.  #점찍으면 플로트(실수)표현
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

# ####스케일링 2 >>>>>>>>>>>>>0~255 → -1~1 변경 #############
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5
# print(np.max(x_train), np.min(x_train)) #1.0 -1.0
# print(np.max(x_test), np.min(x_test))   #1.0 -1.0

x_train = x_train.reshape(-1, 28,28,1)
x_test = x_test.reshape(-1, 28,28,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

# exit()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)


#2. 모델구성
model = Sequential()
model.add(Conv2D(64,(3,3), input_shape=(28,28,1)))  #26,26,64
model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))  #(24,24,32)
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2,), activation='relu')) #(23,23,32)
model.add(Conv2D(16,(2,2,), activation='relu')) #(22,22,16)
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2,), activation='relu')) #(21,21,32)
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2,), activation='relu')) #(20,20,16)

model.add(Flatten())
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape =(32,),activation='relu'))

model.add(Dense(10,activation='softmax'))   #(10,)
model.summary()




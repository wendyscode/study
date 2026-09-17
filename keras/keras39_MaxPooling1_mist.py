#1.mnist 2.cifar10 3.cifar100
#38-0 카피 

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
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
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

x_train = x_train.reshape(-1, 28,28,1)
x_test = x_test.reshape(-1, 28,28,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(10,(2,2),input_shape =(28,28,1),   #10, 10, 10
                 strides=1,                      
                padding='same',                     #너의빈공간을 0으로 채운다
))
model.add(MaxPooling2D())

model.add(Conv2D(filters=9, kernel_size=(3,3),      #8,8,9
                 strides=1,                         
                padding='valid',                    #디폴트 
))
# model.summary()
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

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


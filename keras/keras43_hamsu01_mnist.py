# cnn 모델을 함수형으로 리폼 

#36-2 카피

import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model 
from tensorflow.keras.layers import Conv2D, Dense,Dropout,Flatten,Input
import time
from sklearn.metrics import accuracy_score

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


# #2. 모델구성
# model = Sequential()
# model.add(Conv2D(64,(3,3), input_shape=(28,28,1)))  #26,26,64
# model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))  #(24,24,32)
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,2,), activation='relu')) #(23,23,32)
# model.add(Conv2D(16,(2,2,), activation='relu')) #(22,22,16)
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,2,), activation='relu')) #(21,21,32)
# model.add(Dropout(0.2))
# model.add(Conv2D(16,(2,2,), activation='relu')) #(20,20,16)
# model.add(Flatten())                        #납작하게 펴기📦➡️📏
# model.add(Dense(units=32,activation='relu')) #👉 뉴런 32개를 만들겠다
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape =(32,),activation='relu'))
# model.add(Dense(10,activation='softmax'))   #(10,)
# model.summary()

# exit()

#함수형 
input1 = Input(shape=(28,28,1))
dense1 = Conv2D(filters=64,kernel_size=(3,3), activation='relu',name='ys1')(input1) 
# dense1 = Conv2D(10,(3,3), activation='relu',name='ys1')(input1)  #model.add(Dense(10, input_shape=(3,)))
drop1 = Dropout(0.2)(dense1)            #model.add(Dropout(0.2))
dense2 = Conv2D(9,kernel_size=(3,3),name='ys2')(drop1)     #model.add(Dense(9))
drop2 = Dropout(0.2)(dense2)              
flat = Flatten()(drop2)
output1 = Dense(10,activation='softmax')(flat)               #model.add(Dense(1))
model = Model(inputs=input1, outputs=output1)  

#3. 컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam',
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
y_test = np.argmax(y_test,axis=1,)#.reshape(-1,1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')

'''
cpu
==============model.evaluate======================
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - acc: 0.9890 - loss: 0.0508     
loss : 0.05077488347887993
loss : 0.9890000224113464
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
accuracy_score :  0.989
걸린시간 :  598.45 초

gpu
==============model.evaluate======================
313/313 [==============================] - 1s 2ms/step - loss: 0.0469 - acc: 0.9898
loss : 0.04686978831887245
loss : 0.989799976348877
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9898
걸린시간 :  149.99 초
'''
#0.995맞추기
'''
 cpu 
 1: accuracy_score :  0.9901
걸린시간 :  246.54 초

accuracy_score :  0.9912
걸린시간 :  974.99 초

accuracy_score :  0.9897
걸린시간 :  493.49 초
'''




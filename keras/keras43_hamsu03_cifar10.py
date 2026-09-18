#36-5 카피


from keras.datasets import cifar10
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential,Model 
from tensorflow.keras.layers import Conv2D, Dense,Dropout,Flatten, MaxPooling2D, BatchNormalization,Input,GlobalAveragePooling2D  #🧀
import time
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)   
print(np.max(x_train), np.min(x_train)) 
print(np.max(x_test), np.min(x_test))   

#스케일링 1 
x_train = x_train/255. 
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) 
print(np.max(x_test), np.min(x_test))  

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(50000,1)
y_test = y_test.reshape(10000,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)#

# x_train = x_train.reshape(-1, 32 * 32 * 3)#🍧🍧🍧🍧🍧🍧
# x_test = x_test.reshape(-1, 32 * 32 * 3)    #🍧🍧🍧🍧🍧🍧

#2 모델구성

########################################################🩶🩶🩶
#2-1 함수형 모델구성 
input1 = Input(shape=(32,32,3))
dense1 = Conv2D(filters=64,kernel_size=(3,3), activation='relu',name='ys1')(input1) 
# dense1 = Conv2D(10,(3,3), activation='relu',name='ys1')(input1) 
drop1 = Dropout(0.2)(dense1)            #model.add(Dropout(0.2))
dense2 = Conv2D(9,kernel_size=(3,3),name='ys2')(drop1)     
drop2 = Dropout(0.2)(dense2)              
gap = GlobalAveragePooling2D()(drop2)
output1 = Dense(10,activation='softmax')(gap)             
model = Model(inputs=input1, outputs=output1)  

########################################################🩶🩶🩶

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
model.fit(x_train,y_train, epochs=50000, batch_size=128,
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

y_predict = np.argmax(y_predict,axis=1,)
y_test = np.argmax(y_test,axis=1,)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')


'''
accuracy_score :  0.7062
걸린시간 :  174.36 초

dnn
accuracy_score :  0.5197
걸린시간 :  191.78 초
'''

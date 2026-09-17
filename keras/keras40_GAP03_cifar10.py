#36-5 카피


from keras.datasets import cifar10
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense,Dropout,Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, BatchNormalization#🧀 #🧀
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
y_test = ohe.fit_transform(y_test)#

x_train = x_train.reshape(-1, 32 * 32 * 3)#🍧🍧🍧🍧🍧🍧
x_test = x_test.reshape(-1, 32 * 32 * 3)    #🍧🍧🍧🍧🍧🍧

#2. 모델구성
model = Sequential()
model.add(Dense(128,input_shape = (32*32*3,),activation='relu'))#🍧🍧🍧🍧🍧🍧
model.add(Dense(64,activation='relu'))#🍧
model.add(Dense(32,activation='relu'))#🍧
model.add(Dense(10,activation='softmax'))#🍧
# exit()

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
model.fit(x_train,y_train, epochs=500, batch_size=128,
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

gap 
accuracy_score :  0.6595
걸린시간 :  149.23 초

dnn
accuracy_score :  0.493
걸린시간 :  48.46 초
'''

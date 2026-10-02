#dnn reshape하기 
# LSTM으로 하기 (기존꺼보다 능가하기)#36-2 카피

import pandas as pd#dnn reshape하기 
# LSTM으로 하기 (기존꺼보다 능가하기)#36-2 카피

import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Conv2D,Flatten
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

# x_train = x_train.reshape(-1, 28, 28)
# x_test = x_test.reshape(-1, 28, 28)
# print(x_train.shape, x_test.shape)  

# exit()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)

from tensorflow.keras.layers import Reshape

print(x_train.shape)    #(60000, 28, 28)
from tensorflow.keras.layers import Reshape

model = Sequential()
# (N, 28, 28) → (N, 28, 28, 1)
model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(Conv2D(64, (3,3), activation='relu'))
# (N, 28, 28, 1) → (N, 26, 26, 64)
model.add(Conv2D(32, (3,3), activation='relu'))
# (N, 26, 26, 64) → (N, 24, 24, 32)
# Conv2D 결과를 LSTM이 받을 수 있게 변경
model.add(Reshape((24*24, 32)))
# (N, 24, 24, 32) → (N, 576, 32)

model.add(LSTM(10))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()



# exit()

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=40,
        verbose=1,
        restore_best_weights=True,
)

start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train,
                callbacks=[es, ], verbose=1,
                  epochs=10, batch_size=32,)
end_time = time.time() # 현재시간을 반환. 끝시간


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


import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)
print(x_train.shape, x_test.shape)  

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
model.add(
    LSTM(
        128,
        input_shape=(28, 28),
        return_sequences=True
    )
)
model.add(Dropout(0.2))
model.add(
    LSTM(64)
)
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()


# exit()

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=40,
        verbose=1,
        restore_best_weights=True,
)

start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train,
                callbacks=[es, ], verbose=1,
                  epochs=10, batch_size=32,)
end_time = time.time() # 현재시간을 반환. 끝시간


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




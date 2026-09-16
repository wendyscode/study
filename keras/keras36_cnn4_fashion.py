
from tensorflow.keras.datasets import fashion_mnist

#실습 맹그러봐요 
# acc : 0.92 이상 
# cpu 추천 
  
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense,Dropout,Flatten
import time
from sklearn.metrics import accuracy_score

#1. 데이터 
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)
print(np.max(x_train),np.min(x_train))  #255 0
print(np.max(x_test),np.min(x_test))    #255 0

##### 스케일링 1 >>>>>>>>>>>0~255 → 0~1 변경 ###########
x_train = x_train/255.  #점찍으면 플로트(실수)표현
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)


#2. 모델 구성 
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(Conv2D(16,(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2),activation='relu'))
model.add(Flatten())     

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape = (32,), activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#3.컴파일 훈련 
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

y_predict = np.argmax(y_predict,axis=1,)#.reshape(-1,1)
y_test = np.argmax(y_test,axis=1,)#.reshape(-1,1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')

'''
accuracy_score :  0.906
걸린시간 :  196.05 초
'''




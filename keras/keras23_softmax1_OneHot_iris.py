import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_iris()
print(datasets) #target = y
print(datasets.DESCR) # 묘사하다
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(150, 4) (150,)
print(y)
print(np.unique(y,return_counts=True)) #(array([0, 1, 2]), array([50, 50, 50]))

"""
[0,0,1,1,2] #(5,)
->
[[1,0,0]
[1,0,0]
[0,1,0]
[0,1,0]
[0,0,1]]    #(5,3)

"""
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

# exit()
x_train , x_test, y_train , y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state= 333,  
    shuffle= True,
    stratify=y,  
)
print(x_train.shape,x_test.shape)   #(120, 4) (30, 4)
print(x_train.shape,y_test.shape)   #(120, 4) (30, 3)

# exit()
#2. 모델구성 
model = Sequential()
model.add(Dense(10, input_dim =4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련  
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience= 20,
    restore_best_weights= True,
)
start_time = time.time()
model.fit(x_train,y_train, epochs=1000, batch_size=8,
          verbose=1,
          validation_split = 0.2,
          callbacks =[es]
          )
end_time = time.time()

#4. 평가, 예측 
result = model.evaluate(x_test,y_test,)
print('loss: ',result[0])
print('acc: ',round(result[1],2))

y_predict = model.predict(x_test)

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy_score)
print(" 걸린시간 : ", round(end_time - start_time,2),"초")










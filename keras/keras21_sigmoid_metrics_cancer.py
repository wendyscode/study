import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.datasets import load_breast_cancer #유방암 관련 데이터셋 불러오기

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)   #행 569 열 30
print(datasets.feature_names)

# x = datasets.data
x = datasets['data']
y = datasets.target

print(x.shape,y.shape)  #(569, 30) (569,)
print(type(x))          #<class 'numpy.ndarray'>

print(y)
#0과 1의 개수가 몇개인지 찾아보기. - numpy 
print(np.unique(y))     #[0 1]#y 안에 있는 값들을 중복 없이 한 번씩만 보여줘!
print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])) #뭐가 있는지 + 각각 몇 개인지 알려줘
#0과 1의 개수가 몇개인지 찾아보기. - pandas
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,    
    # test_size=0.7,
    #shuffle=True,
    random_state=337,
      stratify=y,   #데이터를 나눌 때 y의 비율을 똑같이 유지하라
)
print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([148, 250]))
print(np.unique(y_test, return_counts=True))
#(array([0, 1]), array([ 64, 107]))

print(x_train.shape, x_test.shape)  #(398, 30) (171, 30)
print(y_train.shape, y_test.shape)  #(398,) (171,)

#2.모델구성
model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3.컴파일 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',
            # metrics = ['accuracy'],#  몇 % 맞혔는지 확인
            metrics = ['acc'], #애큐러시 몇프로 맞혔는지 확인 

              )
es = EarlyStopping(
    monitor='val_loss',
    mode= 'min',
    patience= 20,
    restore_best_weights= True,
)

start_time = time.time()
hist = model.fit(x_train, y_train , epochs=1000, batch_size=32,
                 verbose = 1,
                 callbacks = [es],
                 validation_split = 0.3,
          )

end_time = time.time()

#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print("=======================================")
print('loss :', loss[0])
print('acc : ', round(loss[1],4))   #소수넷째자리까지 반올림
print("=======================================")

y_pred = model.predict(x_test)
# print(y_pred[:10])
y_pred = np.round(y_pred)
# print(y_pred[:10])

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,y_pred)
print("acc_score :",acc_score)

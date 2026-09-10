# 1에서 원핫 때리고 > 0과 1확인 마지막에 시그모이드대신 소프트맥스 
#로스 : 카테고리컬크로스엔트로피 / 평가지표 np argmax
#시그모이드햇을때와 비교하기 

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
path = 'c:/study/_data/kaggle_santander/'

train_csv = pd.read_csv(path + "train.csv",index_col=0)
test_csv = pd.read_csv(path +'test.csv',index_col=0)
submission_csv = pd.read_csv(path +'sample_submission.csv',index_col=0)

x = train_csv.drop(['target'],axis=1)
y = train_csv['target']
print(x.shape,y.shape)  
#(200000, 200) (200000,)

####################판다스를 넘파이로 바꾸기 ########################
#import numpy as np
y = np.array(y) # 판다스로 땡겨온거 넘파이로 바꾸기 
# y = y.to_numpy()


# exit()
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)

x_train , x_test , y_train , y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True,
    stratify=y,
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 200, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일 훈련
model.compile (loss='categorical_crossentropy', optimizer = 'adam',metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=3,
    restore_best_weights= True,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=64,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es],
          )
end_time = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1],2))

y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test,axis=1)
print(y_test)

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy_score)
print('걸린시간 : ', round(end_time - start_time),'초')

print(submission_csv)
y_pred = model.predict(test_csv)
submission_csv['target'] = y_pred

submission_csv.to_csv(path + "submit/" + "submit_0910_1044.csv")
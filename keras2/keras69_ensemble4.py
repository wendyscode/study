# 69-3카피


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

#1. 데이터 
x1_datasets = np.array([range(100),range(301,401)]).T       #(100,2)
                    #삼성 종가          하이닉스의 종가

y1 = np.array(range(3001,3101))
                    #화성의 화씨 온도 
y2 = np.array(range(13001,13101))    #비트코인 가격 

#맹그러봐!! x1 , x2 삭제
# 단일모델 하나 준비 모델합치는부분에서 인풋에서 데려와라 예측값수정하기 

x_train, x_test,  y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets,
    y1,
    y2,
    test_size=0.2,
    random_state=123
)

#2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10,activation='relu', name='han1')(input1)
dense2 = Dense(20,activation='relu', name='han2')(dense1)
dense3 = Dense(30,activation='relu', name='han3')(dense2)
output1 = Dense(40,activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

#2-3 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output21], name="mg1")
merge1 = Concatenate(name="mg1")([input1, output1])
merge2 = Dense(10, name='mge2')(merge1)
merge3 = Dense(5, name='mge3')(merge2)

#2-5 분기1.
last_dense1 = Dense(10, name = 'ld1')(merge3)
last_dense2 = Dense(10, name = 'ld2')(last_dense1)
last_output1 = Dense(1,name='last1')(last_dense2)

#2-6 분기2.
last_output2 = Dense(1,name='last2')(merge3)

#모델구성
model = Model(inputs = [input1],outputs =[last_output1,last_output2])
# model.summary()

#3. 컴파일 , 훈련 
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train,[y1_train, y2_train],epochs=100,batch_size=8)

#4. 평가예측 
result = model.evaluate(x_test, [y1_test, y2_test])
print('loss : ', result)

x1_pred = np.array([range(100,106),range(400,406)]).T
x2_pred = np.array([range(200,206),range(510,516),
                    range(249,255)]).T
x3_pred = np.array([range(100,106),range(400,406),range(77,83),range(33,39)]).T

y_pred = model.predict(x1_pred)

print("예측값 :")
print(y_pred)


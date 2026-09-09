#  9-1 카피
# verbose :말이많은  => 훈련많이하는거보기싫어 (0번) / 하지만 훈련은 된다 !!


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10]) 
 
#2.모델
model = Sequential()
model.add(Dense(1, input_dim=1))

#3.컴파일 , 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=50,batch_size=10,
          verbose=-1,
          )

# verbose = 0 : 침묵
# verbose = 1 : 디폴트 (다보여줘)
# verbose = 2 : 프로그래스바 삭제
# verbose = 나머지(0,1,2을제외) : 에포만 나옴.

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)


import random
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__)

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터 
x = np.array([1,])
y = np.array([1,])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
print("====================model.weights====================")
print(model.weights)
print("====================model.trainable.weights====================")
print(model.trainable_weights)
print("====================================")
print(len(model.weights))   #6
print(len(model.trainable_weights)) #6

####################### 동결 ##############################
# model.trainable = False #동결 ★★★★★
model.trainable = True  #동결안함
####################### 동결 ##############################
# model.summary()


# exit()
#3. 컴파일 , 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, batch_size = 1, epochs= 1000 , verbose = 1,)

#4. 평가, 예측 
y_predict = model.predict(x)
print(y_predict)


print("====================================================")
print(model.weights)
print("====================================================")


#동결X

#동결O
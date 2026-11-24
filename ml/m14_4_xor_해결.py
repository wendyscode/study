import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)

#1. 데이터 
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])

#2. 모델 
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Dense(1 , activation='sigmoid'))

#3. 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_data, y_data, epochs=100)

#4 평가, 예측 
y_predict = model.predict(x_data)

result = model.evaluate(x_data, y_data)
print("loss,acc: ", result)

acc = accuracy_score(y_data, y_predict.round())
print("accuracy_score : ", acc)

acc = accuracy_score(y_data, np.round(y_predict))
print("accurrcy_score:",acc)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import random
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

##### 스케일링 : 0~255 -> 0~1 #####
x_train = x_train / 255.
x_test = x_test / 255.

print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))


# 2. 모델 
# model = VGG16()
########### 디폴트 ############
vgg16 = VGG16(
    weights= 'imagenet', 
    include_top= False,
    input_shape=(32, 32, 3),    
)

# vgg16.trainable = False     # VGG16 가중치 동결O
vgg16.trainable  = True    # VGG16 가중치 동결X

#vgg16.summary()

model1 = Sequential()
model1.add(vgg16)
model1.add(Flatten())
model1.add(Dense(100))
model1.add(Dense(100))
model1.add(Dense(10, activation='softmax'))

# model.summary()

# 3. 컴파일, 훈련
learning_rate = 0.001

model1.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['acc']
)

es1 = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)

start_time1 = time.time()

model1.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[es1]
)

end_time1 = time.time()


# 4. 평가, 예측
print('============== 동결 O model.evaluate ======================')
loss1 = model1.evaluate(x_test, y_test, verbose=1)

print('loss :', loss1[0])
print('acc :', loss1[1])

y_predict1 = model1.predict(x_test)
y_predict1 = np.argmax(y_predict1, axis=1)

acc_score1 = accuracy_score(y_test, y_predict1)

print('동결 O accuracy_score :', acc_score1)
print('동결 O 걸린시간 :', round(end_time1 - start_time1, 2), '초')

# print('동결 X accuracy_score :', acc_score1)
# print('동결 x 걸린시간 :', round(end_time1 - start_time1, 2), '초')


#실습#
# 비교할거 
# #1. 이전에 본인이 한 최상의 결과 . 
# 2. 가중치를 동결하고 훈련시켰을때,  trainable= False
# 3. 가중치를 동결하지 않고 훈련 시켰을때, trainable= True
# 4. 시간 까지 비교 

#동결o
# 동결 O accuracy_score : 0.5836
# 동결 O 걸린시간 : 46.94 초

#동결x
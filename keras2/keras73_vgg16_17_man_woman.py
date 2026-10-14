import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import random
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16

# 1. 데이터 (keras47_03_load_npy_men_women.py 와 동일하게 로드 후 분할)
x_data = np.load('./_save/keras46/manwoman_x.npy')
y_data = np.load('./_save/keras46/manwoman_y.npy')

print(x_data.shape, y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8, shuffle=True, random_state=123,
    stratify=y_data
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(np.max(x_train), np.min(x_train))   # 스케일링 안 돼 있으면 255 0 -> 아래 두 줄 주석 해제
# x_train = x_train / 255.
# x_test = x_test / 255.


# 2. 모델 (동결 O / 동결 X 를 한 번에 비교)
def run(trainable, learning_rate):
    ########### 디폴트 ############
    vgg16 = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=x_train.shape[1:],   # npy 데이터 shape 그대로 사용
    )
    vgg16.trainable = trainable          # False : 동결 O / True : 동결 X

    model = Sequential()
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(1, activation='sigmoid'))   # 남/여 이진 분류

    # 3. 컴파일, 훈련
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['acc']
    )

    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True
    )

    start_time = time.time()
    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=1,
        validation_split=0.2,
        callbacks=[es]
    )
    end_time = time.time()

    # 4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)
    print('loss :', loss[0])
    print('acc :', loss[1])

    y_predict = model.predict(x_test)
    y_predict = np.round(y_predict).reshape(-1)

    acc_score = accuracy_score(y_test, y_predict)
    return acc_score, round(end_time - start_time, 2)


print('============== 동결 O ======================')
acc1, time1 = run(trainable=False, learning_rate=0.001)

print('============== 동결 X ======================')
acc2, time2 = run(trainable=True, learning_rate=0.0001)   # 동결 X 는 lr 을 낮춰야 안 깨짐

print('동결 O accuracy_score :', acc1)
print('동결 O 걸린시간 :', time1, '초')
print('동결 X accuracy_score :', acc2)
print('동결 X 걸린시간 :', time2, '초')


#실습#
# 비교할거
# 1. 이전에 본인이 한 최상의 결과 .
# 2. 가중치를 동결하고 훈련시켰을때,  trainable= False
# 3. 가중치를 동결하지 않고 훈련 시켰을때, trainable= True
# 4. 시간 까지 비교

#이전 최상의 결과 (keras47_03_load_npy_men_women.py)
# accuracy_score :
# 걸린시간 :

#동결o
# 동결 O accuracy_score :
# 동결 O 걸린시간 :

#동결x
# 동결 X accuracy_score :
# 동결 X 걸린시간 : 
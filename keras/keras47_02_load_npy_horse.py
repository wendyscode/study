
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

#1.데이터

np_path = './_save/keras46/'
x_train = np.load(np_path + 'keras46_01_x_train_horse-human.npy')
y_train = np.load(np_path + 'keras46_01_y_train_horse-human.npy')
x_test = np.load(np_path + 'keras46_01_x_test_horse-human.npy')
y_test = np.load(np_path + 'keras46_01_y_test_horse-human.npy')


# 2. 모델 구성 (CNN)
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(124, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# 3. 컴파일 및 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                    restore_best_weights=True,
                    verbose=1, #🤎
                    )

# 3.컴파일 훈련
# generator 데이터셋 구조에 맞는 fit 실행
start_time = time.time()
hist = model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[es],
    verbose=1, #🤎
)
end_time = time.time()
 # 4. 평가


print("==============model.evaluate======================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('loss :', loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')

'''
accuracy_score :  0.8116658428077114
걸린시간 :  230.22 초

accuracy_score :  0.8126544735541276
걸린시간 :  52.2 초
'''
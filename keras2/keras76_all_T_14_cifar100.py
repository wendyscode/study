import warnings
warnings.filterwarnings("ignore")   # 경고없애는거임

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar100
from keras.applications import VGG19, ResNet50, ResNet101, DenseNet201, MobileNetV2, EfficientNetB0

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터 (반복문 밖에서 한 번만)
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 100

# 2. 모델 6개 (클래스만 넣고 반복문 안에서 생성)
model_list = [VGG19, ResNet50, ResNet101, DenseNet201, MobileNetV2, EfficientNetB0]

results = []

for model_cls in model_list:
    tf.keras.backend.clear_session()

    model_i = model_cls(include_top=False, input_shape=IMG_SHAPE)
    model_i.trainable = False

    model = Sequential()
    model.add(Input(shape=IMG_SHAPE))
    # 스케일링 (EfficientNet은 내부에서 정규화하므로 0~255 그대로)
    if model_cls is not EfficientNetB0:
        model.add(Rescaling(1./255))
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    # 클래스가 100개라 32,16이면 병목이 너무 심해서 노드 수를 키움
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # 3. 컴파일 훈련
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])

    es = EarlyStopping(monitor='val_loss', mode='min',
                       patience=10, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', mode='min',
                            patience=5, factor=0.5)

    model.fit(x_train, y_train, epochs=1000, batch_size=128,
              verbose=0,
              validation_split=0.2,
              callbacks=[es, rlr])

    # 4. 평가 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    results.append((model_i.name, loss[0], loss[1]))
    print("==============cifar 100===================")
    print("모델명 : ", model_i.name, 'loss : ', loss[0], ' acc : ', loss[1])

print("\n========== cifar100 최종 결과 ==========")
for name, l, a in results:
    print(f"{name:20s} loss : {l:.4f}   acc : {a:.4f}")
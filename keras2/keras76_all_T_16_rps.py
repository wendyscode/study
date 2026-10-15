import warnings
warnings.filterwarnings("ignore")   # 경고없애는거임

import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import VGG19, ResNet50, ResNet101, DenseNet201, MobileNetV2, EfficientNetB0

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터
# ★ 경로를 본인 폴더에 맞게 수정하세요 (안에 클래스별 하위 폴더: paper/, rock/, scissors/)
path = './_data/image/rps/'
IMG_SIZE = (100, 100)
IMG_SHAPE = IMG_SIZE + (3,)

ds = tf.keras.utils.image_dataset_from_directory(
    path, image_size=IMG_SIZE, batch_size=256,
    label_mode='int', shuffle=False)
class_names = ds.class_names
NUM_CLASSES = len(class_names)
print("클래스 :", class_names)

x, y = [], []
for xb, yb in ds:
    x.append(xb.numpy().astype('uint8'))   # 메모리 절약을 위해 uint8로 저장
    y.append(yb.numpy())
x = np.concatenate(x)
y = np.concatenate(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333, stratify=y)
print(x_train.shape, x_test.shape)

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
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # 3. 컴파일 훈련
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])

    es = EarlyStopping(monitor='val_loss', mode='min',
                       patience=10, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', mode='min',
                            patience=5, factor=0.5)

    model.fit(x_train, y_train, epochs=1000, batch_size=32,
              verbose=0,
              validation_split=0.2,
              callbacks=[es, rlr])

    # 4. 평가 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    results.append((model_i.name, loss[0], loss[1]))
    print("===============rps==================")
    print("모델명 : ", model_i.name, 'loss : ', loss[0], ' acc : ', loss[1])

print("\n========== rps 최종 결과 ==========")
for name, l, a in results:
    print(f"{name:20s} loss : {l:.4f}   acc : {a:.4f}")
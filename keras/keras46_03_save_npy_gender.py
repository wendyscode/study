# keras46_03_save_npy_men_women.py

# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. ImageDataGenerator 정의 (스케일링 및 데이터 증강 설정 가능)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # 필요 시 가로 뒤집기, 회전 등 증강 조건 추가
)

# 2. 폴더에서 이미지 데이터 불러오기
# 데이터 경로 예시: './data/men_women/' 하위에 'men' 폴더와 'women' 폴더가 존재해야 합니다.
xy_train = train_datagen.flow_from_directory(
    './_data/man_woman/',
    target_size=(64, 64),  # 이미지 크기 통일
    batch_size=30000,         # 전체 데이터를 한 번에 가져오기 위해 큰 값 설정
    class_mode='binary',     # 남성/여성 이진 분류 (0 또는 1)
    shuffle=True,
)

print("총 이미지 개수 (samples):", xy_train.samples)

# 3. 데이터 추출
x_data = xy_train[0][0]
y_data = xy_train[0][1]

print("x_data 형태:", x_data.shape) # (5000, 150, 150, 3)
print("y_data 형태:", y_data.shape) # 예: (5000,)

#  NumPy 파일(.npy)로 저장하기
# 이후 코드(keras46_04_load_npy_men_women 등)에서 호출하여 빠르게 사용합니다.
np.save('./_save/keras46/manwoman_x.npy', arr=x_data)
np.save('./_save/keras46/manwoman_y.npy', arr=y_data)

print("--- npy 파일 저장 완료 ---")
print("class_indices:", xy_train.class_indices)

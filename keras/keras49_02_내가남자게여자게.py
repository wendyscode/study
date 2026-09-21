# keras49_02_내가남자게여자게.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image

#.1 내 사진 이미지 로드 및 전처리
img_path = './_data/image/내사진.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0  
img_array = np.expand_dims(img_array, axis=0) # (1, 150, 150, 3)으로 차원 확장

# 2. 모델 구조 정의 (2번 파일 keras47_03과 100% 똑같은 구조여야 가중치가 에러 없이 로드됩니다!)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # 이진 분류 출력층
])
#  저장해 둔 가중치 불러오기
model.load_weights('./_save/keras46/manwoman_weights.h5')
print("--- 가중치 로드 완료 ---")

# 4. 내 사진으로 "내가 남자게 여자게" 예측하기
print("============== 내 사진 예측 ======================")

# 4. 예측 수행
y_predict = model.predict(img_array)
print('예측 확률값 (raw):', y_predict[0][0])

# 반올림하여 0 또는 1로 변환
y_result = np.round(y_predict)
print('반올림 결과 (0 or 1):', y_result[0][0])

# 결과 출력
if y_result[0][0] == 0:
    print(f"결과: 남자일 확률이 { (1 - y_predict[0][0]) * 100:.2f}% 입니다. (남자)")
else:
    print(f"결과: 여자일 확률이 { y_predict[0][0] * 100:.2f}% 입니다. (여자)")

# # 실행 예시 (본인 사진 파일명 입력)
# predict_my_gender('./_data/image/내사진.jpg')

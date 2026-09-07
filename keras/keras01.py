import tensorflow as tf # TF로 줄여서 쓸거야 
print (tf.__version__)

from tensorflow.keras.models import Sequential #순차적인 모델을 할거야 
from tensorflow.keras.layers import Dense #댄스 레이어를 가져와 
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential() #dim은 차원이다 = 차원이 1개 = 벡터가 한개 
model.add(Dense(1, input_dim=1)) #한개가 들어가서 한개가 나온다 

#3. 컴파일, 훈련
model.compile(loss='mse' , optimizer='adam')
model.fit(x,y,epochs=100) #몇번돌릴건지 쓰기 

#4. 평가예측
result = model.predict(np.array([4]))
print ("4의 예측값  : ", result)

#4의 예측값  :  [[-4.032362]] / 가장 좋은 결과값 써두기 

#1. 데이터 
#2. 모델구성
#3.컴파일, 훈련
#4. 평가, 예측

#kingkeras@naver.com -> 윤영선 선생님 이메일

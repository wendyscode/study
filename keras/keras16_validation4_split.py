from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

#1.데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(
    x,y,
    # test_size=0.5,
    train_size=0.75,
    # shuffle=False,
    random_state=111,
)

#2.모델
model = Sequential()
model.add(Dense(1, input_dim=1))

#3.컴파일 , 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=10,
          verbose= 1,
        #   validation_data = (x_val , y_val),
          validation_split = 0.33, # 33% 정도를 검증에 사용해주세요!
          )

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)
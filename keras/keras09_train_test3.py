import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[검색] train과 test를 섞어서 7:3 나눈다
# 힌드 : 사이킷런 scikit-learn = sklearn

# 함수의 결과를 앞에다가 저장하라
# train_size와 test_size는 합이 1이 되도록 설정해야 한다.
# (train_size=0.7, test_size=0.3) => 전체 데이터의 70% 학습, 30% 테스트
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.7, #디폴트는0.75/0.25
    # test_size=0.3,  # 하나만 써도된다 
    # shuffle=True,  # 디폴트는 섞는다.
    random_state=333, #랜덤 숫자를 넣는다/ 랜덤숫자에따라 똑같은 숫자가 나옴 
)
print('x_train = ', x_train)
print('x_test = ', x_test)
print('y_train = ', y_train)
print('y_test = ', y_test)

#2.모델구성
model = Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100, batch_size=2)

#4. 평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
results = model.predict(x_test)
print('result : ', results)
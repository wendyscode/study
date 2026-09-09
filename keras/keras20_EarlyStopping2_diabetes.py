# 얼리스타핑 적용하고 더좋앗던 성능으로 갱신시키기
# - 에포 마니주고 
# - 페이션트 크게준다 

#R2 기준 0.62 이상 만들어보기 

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=333)
x_train, x_val , y_train, y_val = train_test_split(x_train, y_train,
                                                   train_size = 0.5,
                                                    random_state=333,)

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=10))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience= 30,
    restore_best_weights= True,
)

hist = model.fit(x_train, y_train, epochs=100, batch_size=2,
          validation_data = (x_val, y_val),
          callbacks = [es],)

#4.평가예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)
# results = model.predict(x_test)
# print(results)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

print("===================history========================")
print(hist)
print("===================hist.history========================")
print(hist.history)
print("===================loss========================")
print(hist.history["loss"])
print("===================val_loss========================")
print(hist.history["val_loss"])
print("===========================================")

####그래프그리기####
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'       #한글깨짐현상 폰트지정해주기 

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:],c='red',label='loss')# y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][2:],c='blue',label='val_loss')
plt.legend(loc= 'upper right')#우측상단에 라벨표시  #location 위치어디로할건지 
plt.title('당뇨병 Loss') 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid() #격자표시 추가 
plt.show()

# 3 애포 부터 100애포 까지 
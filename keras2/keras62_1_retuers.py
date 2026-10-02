from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , LSTM, Embedding,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000, #단어사전의 갯수 , 빈도수가 높은 1000개의 단어순으로 100개 뽑겠다.
    # maxlen=1000,     #단어갯수의 최대길이제한/ 디폴트전체길이
    test_split=0.2, #
)

print(x_train)
print(x_train.shape, y_train.shape) #(8982,) (8982,)
print(x_test.shape, y_test.shape)   #(2246,) (2246,)
print(y_train)                      #[ 3  4  3 ... 25  3 25]
print(np.unique(y_train))              #[ 0  1  2 ... 44 45 46]   #46개의 카테고리  

print(type(x_train))     #<class 'numpy.ndarray'>
print(type(x_train[0]))  #<class 'list'>  #리스트안에 정수형태의 단어인덱스가 들어있음  
print(len(x_train[0]),len(x_train[1]))  #87,56

print("뉴스기사의 최대길이:",max(len(i) for i in x_train))  #2376
print("뉴스기사의 최소길이:",min(len(i) for i in x_train))  #13
print("뉴스기사의 평균길이:",sum(map(len,x_train))/len(x_train))#145.53

# # # # # ## #  # # # 전처리(패드 시쿼스)# # # # # # # # # # # # # # # # # # # 
max_len = 200         # 모든 뉴스 기사의 길이를 200으로 맞춤
x_train = pad_sequences(
    x_train,
    maxlen=max_len,
    padding='post',
    truncating='post'
)
x_test = pad_sequences(
    x_test,
    maxlen=max_len,
    padding='post',
    truncating='post'
)
print("패딩 후:", x_train.shape)    # (8982, 200)
print("패딩 후:", x_test.shape)     #(2246, 200)

## # # # # # # # # # # #  y 원핫 # # # # # # # # # # # # # # # # # # # # 
num_classes = 46
y_train = to_categorical(
    y_train,
    num_classes=num_classes
)
y_test = to_categorical(
    y_test,
    num_classes=num_classes
)
print("y_train:", y_train.shape)     #(8982, 46)
print("y_test :", y_test.shape)     #(2246, 46)

#2. 모델구성 
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64,
        input_length=max_len))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3)) # 과적합 방지
model.add(Dense(num_classes,activation='softmax'))# 다중분류

#3. 컴파일, 훈련
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',# 다중분류 + 원핫 인코딩
    metrics=['accuracy']    # 평가지표
)
# model.summary()   

es = EarlyStopping(
    monitor='val_loss', 
    patience=10,     
    mode='min',        
    restore_best_weights=True 
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=128,
    verbose=1,
    callbacks=[es],
)

#4. 평가, 예측
loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print()
print("==============================")
print("Test Loss     :", loss)
print("Test Accuracy :", accuracy)
print("==============================")

if accuracy >= 0.67:
    print("Accuracy 0.67 이상 달성!")
else:
    print("Accuracy 0.67 미달성")

#다중분류 카테고리컬 크로스엔트로피 / 임베딩다음 LSTM붙이기 /바이디렉셔널붙여도상관없음
#평가지표는 로스와 에큐러시/ 에큐러시 0.67 이상 나오기 

'''
==============================
Test Loss     : 1.0410881042480469
Test Accuracy : 0.7631344795227051
==============================
Accuracy 0.67 이상 달성!
'''
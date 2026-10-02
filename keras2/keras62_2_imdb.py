from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , LSTM, Embedding,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000, #단어사전의 갯수 , 빈도수가 높은 1000개의 단어순으로 100개 뽑겠다.
    # maxlen=1000,     #단어갯수의 최대길이         
    # test_split=0.2, #
)

# 맹그러봐!!! 
# acc기준 0.6
print(x_train)  
print(y_train)
print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(x_test.shape, y_test.shape)   # (25000,) (25000,)
print(np.unique(y_train)) # [0 1]

print("최대길이:",max(len(i) for i in x_train))  # 2494
print("최소길이:",min(len(i) for i in x_train))  # 11
print("평균길이:",sum(map(len,x_train))/len(x_train))#238.71364

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
print("패딩 후:", x_train.shape)    #  (25000, 200)
print("패딩 후:", x_test.shape)     # (25000, 200)

#2. 모델구성 
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64,
        input_length=max_len))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3)) # 과적합 방지
model.add(Dense(1, activation='sigmoid')) # 2진 분류이므로 노드 1개, sigmoid!

#3. 컴파일, 훈련
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # 2진 분류용 손실 함수
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

if accuracy >= 0.60:
    print("Accuracy 0.60 이상 달성!")
else:
    print("Accuracy 0.60 미달성")

'''
==============================
Test Loss     : 0.3795115649700165
Test Accuracy : 0.8342000246047974
==============================
Accuracy 0.60 이상 달성!
'''



# =====================================================

# 1. 단어 사전(정수와 단어의 짝꿍) 가져오기
word_index = imdb.get_word_index()

# 2. 케라스 내부적으로 사용하는 특수 인덱스 맞추기 (패딩, 시작 토큰 등 때문에 3을 밀어줍니다)
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # 사전에 없는 단어
word_index["<UNUSED>"] = 3

# 3. 숫자를 다시 단어로 바꿔주는 거꾸로 사전 만들기 (정수 -> 단어)
reverse_word_index = {v: k for k, v in word_index.items()}

# 4. 첫 번째 리뷰의 숫자들을 단어로 변환해서 합치기!
decoded_review = " ".join([reverse_word_index.get(i, "?") for i in x_train[0]])

print("--- 실제 영어 리뷰 내용 ---")
print(decoded_review)
print("\n--- 이 리뷰의 정답 (0: 부정, 1: 긍정) ---")
print(y_train[0])

# Embedding + LSTM/ GRU
# Embedding + Bidirectional
# Embedding + Flatten + DNN
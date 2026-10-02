import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#1. 데이터 
docs = [
    '너무 재미있다.', '참 최고에요.', '참 잘만든 영화예요.',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요.', '글쎄',
    '별로예요.', '생각보다 지루해요.', '연기가 어색해요.',
    '재미없어요.', '너무 재미없다.', '참 재밌네요.',
    '개똥이 바보','말똥이 잘생겼다','길동이 또 구라친다',
]
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]) # 긍정:1 부정:0

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index) # 단어사전 확인
#{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6,
#  '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, 
# '보고': 13, '싶네요': 14, '글쎄': 15, '별로예요': 16, '생각보다': 17, '지루해요': 18,
#  '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23,
#  '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)

# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16],
#  [17, 18], [19, 20, 21], [2, 22],[1, 23], [24, 25], [26, 27], [28, 29, 30]]


############ 패딩 ####################
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                          padding='pre', # pre: 앞에 0을 채움, 뒤에'post'는 뒤에 0을 채움
                          maxlen=5,       #maxlen: 최대길이 #디폴트는 앞에짤림!
                          truncating='post' # pre: 앞에서부터 자름, 뒤에'post'는 뒤에서부터 자름
) 

print(padded_x)
print(padded_x.shape) # (15, 5)    
#맹그러봐!! #y는 라벨s / 원핫 하지마/ 이진분류 시그모이드 / acc목표1.0
# x=15.5 y=15, dnn구성 /트레인테스트분리 5개정도 별도로 빼기

x_train, x_test, y_train, y_test = train_test_split(
    padded_x,
    labels,
    test_size=5,
    random_state=42,
    stratify=labels
)
print(x_train.shape) # (10, 5)
print(x_test.shape)  # (5, 5)

#2. 모델구성
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(5,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(                  # 학습
    x_train,
    y_train,
    epochs=1000,
    batch_size=2,
    verbose=1
)

#4. 평가, 예측
loss, acc = model.evaluate(         # 테스트
    x_test,
    y_test,
    verbose=0
)

print("test loss :", loss)
print("test accuracy :", acc)

#x_predict = '개똥이 잘생겼다.' 예측해보기  
# 새로운 문장 예측
x = token.texts_to_sequences(['개똥이 잘생겼다.'])

print(x)

x = pad_sequences(
    x,
    padding='pre',
    maxlen=5,
    truncating='post'
)

print(x)
print(x.shape)
y_predict = model.predict(x)
print(y_predict)

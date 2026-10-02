# 61-3 카피 

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

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

#2. 모델 
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

model = Sequential()
#######################임베딩1##############################
model.add(Embedding(input_dim=30, output_dim=100, input_length=5)) 
                #inputdim단어사전갯수 / outputdim차원
model.add(SimpleRNN(10))
model.add(Dense(1))
#  embedding (Embedding)       (None, 5, 100)            3000                                                            
#  simple_rnn (SimpleRNN)      (None, 10)                1110   

#####################임베딩2 ###################################
model.add(Embedding(input_dim=30, output_dim=100)) #input_length명시안해도 알아서 맞춰줌
                #inputdim단어사전갯수 / outputdim차원
model.add(SimpleRNN(10))
model.add(Dense(1))

######################임베딩3 ####################################
model.add(Embedding(30,100)) #순서: imputdim, outputdim
# model.add(Embedding(30,100,5)) #imputdim, outputdim 안된다!!! 
model.add(Embedding(30,100,input_length=5))
model.add(SimpleRNN(10))
model.add(Dense(1))










model.summary()
# exit()
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(padded_x, labels, epochs=3,)


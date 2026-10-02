from tensorflow.keras.preprocessing.text import Tokenizer

text1 = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."
text2 = "개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다."

token = Tokenizer() # 인스턴스(객체) = 클래스(),인스턴스 생성
token.fit_on_texts([text1, text2])

print(token.word_index)

# 맹글기

# 문장을 단어로 나누기
text1 = text1.split()
text2 = text2.split()

import numpy as np
from sklearn.preprocessing import OneHotEncoder
texts = np.concatenate((text1, text2))
print(texts)

texts = texts.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
onehot = encoder.fit_transform(texts)
print(onehot)
print(encoder.categories_)
print(onehot)
from tensorflow.keras.preprocessing.text import Tokenizer

text = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."

token = Tokenizer() # 인스턴스(객체) = 클래스(),인스턴스 생성
token.fit_on_texts([text]) # 단어사전 생성
print(token.word_index) # 단어사전 확인
#{'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '김밥을': 6, '엄청': 7, '먹었다': 8}

print(token.word_counts) # 단어 빈도수 확인
#OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text]) # 단어를 시퀀스(정수)로 변환
print(x) # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
print(len(x)) # 1

################# 원핫 인코딩 3가지 만들기##############
#1. pandas
import pandas as pd
df_pandas = pd.get_dummies(df, columns=['color'])

print(df_pandas)

#2. sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['color']])
print(encoded)

#3. keras
from tensorflow.keras.utils import to_categorical
# red=0, blue=1, green=2
y = [0, 1, 2, 0, 1]
encoded = to_categorical(y, num_classes=3)
print(encoded)


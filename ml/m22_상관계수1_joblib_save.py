import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1.데이터
dataests = load_iris()
print(dataests.feature_names)

x = dataests['data']
y = dataests.target

df = pd.DataFrame(x, columns=dataests.feature_names)

df['Target'] = y

print(df)

print("================== 상관관계 히트맵 ======================")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(),square=True,annot=True, cbar=True)
plt.show()


#맹그러봐
#joblib, pinkle로 세이브하기! 

# 상관계수 계산 및 확인
correlation = df.corr(method='pearson')
print('특성 간 상관계수:')
print(correlation)

# 상관계수 절댓값이 큰 특성 쌍 확인 (자기 자신과의 상관계수는 제외)
correlation_pairs = correlation.where(
    np.triu(np.ones(correlation.shape), k=1).astype(bool)
).stack().sort_values(key=np.abs, ascending=False)
print('상관계수 절댓값 기준 정렬:')
print(correlation_pairs)

# joblib, pickle로 상관계수 결과 저장
import joblib
import pickle
import os

path = os.path.join(os.path.dirname(__file__), '_save', 'joblib')
os.makedirs(path, exist_ok=True)
joblib.dump(correlation, os.path.join(path, 'm22_corr_save.joblib'))
with open('iris_correlation.pkl', 'wb') as file:
    pickle.dump(correlation, file)

print('저장 완료: _save/joblib/m22_corr_save.joblib, iris_correlation.pkl')

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델
model = DecisionTreeClassifier()

# 3. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)
print(f'acc = {scores}, \n 평균 acc = {np.mean(scores):.4f}')
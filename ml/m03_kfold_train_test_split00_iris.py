import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

x_train, x_test, y_train , y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델
model = DecisionTreeClassifier()

# 3. 훈련, 평가
scores = cross_val_score(model,x_train, y_train, cv=kfold)
print(f'acc = {scores}, \n 평균 acc = {np.mean(scores):.4f}')

#acc = [0.93333333 0.9        1.         0.96666667 0.93333333], 
#  평균 acc = 0.9467

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

#맹그러봐 accuracy_score
acc = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy_score(y_test, y_predict):.4f}')

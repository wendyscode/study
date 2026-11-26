# Pseudo Labeling 기법:  모델을 돌려서 나온 결과로 결측치를 찾음
# 스태킹 : 모델을 돌려 나온거로 컬럼을 구성해서 새로운 데이터셋을 만듦
    #    :한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼을 만듦

import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import BaggingRegressor, VotingClassifier, VotingRegressor

#1. 데이터
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test , y_train, y_test = train_test_split(
    x,y,train_size=0.8, random_state=444,
    stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델

xgb = XGBClassifier()
lg = LGBMClassifier(verbosity=-1)
rf = RandomForestClassifier(verbose=0)
cat = CatBoostClassifier()

models = [xgb, lg, cat, rf]
train_list = []
test_list = []

for model in models:
    #3.훈련
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_pred)
    test_list.append(y_test_pred)

    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0}ACC:{1:.4f}'.format(class_name, score))

x_train_new = np.array(train_list).T
print(x_train_new.shape)

x_test_new = np.array(test_list).T
print(x_test_new.shape)

#2-2. 모델
model2 = CatBoostClassifier(verbose=0)

#3-2. 훈련
model2.fit(x_train_new, y_train)

#4. 평가, 예측
y_pred_new = model2.predict(x_test_new)
score_new = accuracy_score(y_test, y_pred_new)
print("스태킹 결과:", score_new)




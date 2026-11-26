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
from sklearn.ensemble import BaggingRegressor, VotingClassifier, VotingRegressor, StackingClassifier

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

model= StackingClassifier(
    estimators=[('XGB', xgb),('LG',lg),('CAT',cat),('RF', rf)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs = -1,
    cv = 5,
)

#3-2. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
score_new = accuracy_score(y_test, y_pred)
print("스태킹 결과:", score_new)

#스태킹 결과: 0.956140350877193



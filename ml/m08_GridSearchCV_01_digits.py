import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.05, 0.01]
}

#2. 모델
xgb = XGBClassifier(
    random_state=333,
    eval_metric='mlogloss'
)
model = GridSearchCV(
    xgb, parameters, cv=kfold,
    verbose=1,
    n_jobs=-1,
    scoring='accuracy'
)

#3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

#4. 평가, 예측
print('model.score:', model.score(x_test, y_test))
print('best_score:', model.best_score_)

print("걸린시간 : ", round(end - start, 2), "초")


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 클래스파이어
# R2 대신 정확도(accuracy)로 평가

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=333,
    stratify=y,
)

# 2. 모델
model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=333),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,  # 복원 추출
    random_state=333,
    n_jobs=-1,
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 및 예측
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('최종점수 (accuracy):', model.score(x_test, y_test))
print('accuracy_score:', accuracy)
print(f'정확도: {accuracy:.4f}')

# BaggingClassifier 옵션 참고
# bootstrap=True  : 복원 추출로 표본을 구성 (배깅)
# bootstrap=False : 비복원 추출로 표본을 구성 (페이스팅)

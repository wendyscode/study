from sklearn.datasets import load_diabetes
#맹그러봐!!

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터 준비
diabetes = load_diabetes()
x, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names
print(x.shape, y.shape)

# 2. 학습 데이터와 테스트 데이터 분리
random_state = 12313
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=random_state,
)

# 3. 모델 구성 및 학습
model = RandomForestRegressor(random_state=random_state)
model.fit(x_train, y_train)

# 4. 평가
score = model.score(x_test, y_test)
print("R2:", score)

# 5. 특성 중요도 출력
feature_importances = model.feature_importances_
order = feature_importances.argsort()[::-1]

for index in order:
    print(f"{feature_names[index]}: {feature_importances[index]:.4f}")
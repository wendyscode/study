import warnings
import numpy as np
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

warnings.filterwarnings("ignore")

# 1. 데이터
x, y = load_iris(return_X_y=True)
print(f"x shape = {x.shape}, y shape = {y.shape}")

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모든 분류 모델 불러오기
estimators = all_estimators(type_filter="classifier")
print("scikit-learn version:", sk.__version__)
print("분류 모델 개수:", len(estimators))

# 3. 모델별 K-fold 교차검증
results = []
for name, estimator_class in estimators:
    try:
        model = estimator_class()
        params = model.get_params(deep=False)
        if "random_state" in params:
            model.set_params(random_state=333)

        # StandardScaler는 매 fold의 훈련 데이터에만 fit되도록 Pipeline으로 묶음
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(
            pipeline, x, y, cv=kfold, scoring="accuracy", n_jobs=-1,
            error_score="raise"
        )
        mean_score = scores.mean()
        std_score = scores.std()
        results.append((name, mean_score, std_score))
        print(
            f"{name:35s} acc = {scores} | "
            f"평균 acc = {mean_score:.4f} (+/- {std_score:.4f})"
        )
    except Exception as e:
        print(f"{name:35s} 모델 오류: {type(e).__name__}: {e}")

# 평균 정확도 기준 상위 모델 출력
if results:
    best_name, best_mean, best_std = max(results, key=lambda row: row[1])
    print("\n평균 정확도 기준 최고 모델")
    print(f"{best_name}: 평균 acc = {best_mean:.4f} (+/- {best_std:.4f})")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. 데이터
DATA_PATH = Path(r"C:\study\_data\kaggle_santander\train.csv")
df = pd.read_csv(DATA_PATH)

# ID_code는 순번 성격의 식별자이므로 학습 특징에서 제외
x = df.drop(columns=["ID_code", "target"])
y = df["target"]
print(f"x shape = {x.shape}, y shape = {y.shape}")
print(f"target 비율 =\n{y.value_counts(normalize=True).sort_index()}")

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델
model = HistGradientBoostingClassifier(random_state=333)

# 3. 훈련, 평가
# 불균형한 이진 분류 데이터이므로 정확도 대신 ROC AUC를 사용
scores = cross_val_score(model, x, y, cv=kfold, scoring="roc_auc", n_jobs=-1)
print(f"roc_auc = {scores}")
print(f"평균 roc_auc = {scores.mean():.4f}")

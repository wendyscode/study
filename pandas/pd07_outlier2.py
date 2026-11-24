import numpy as np
import matplotlib.pyplot as plt


aaa = np.array([
    [-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
    [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350],
]).T

#맹그러봐!!!! 찾아내라!!!!
#이상치 처리해보세요!!!!

# 2차원 데이터에서 열(column)별로 이상치를 찾아보자.
# 각 열의 값 범위가 다르므로, 열마다 IQR 기준을 따로 계산한다.
print('원본 데이터 (행, 열):')
print(aaa)


def find_outliers_by_column(data):
    """열별 IQR 기준 이상치 위치와 기준값을 반환한다."""
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 행/열 위치별로 이상치 여부를 표시하는 2차원 불리언 배열
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    return outlier_mask, q1, q3, iqr, lower_bound, upper_bound


mask, q1, q3, iqr, lower, upper = find_outliers_by_column(aaa)

for col in range(aaa.shape[1]):
    print(f'\n[열 {col}]')
    print(f'Q1: {q1[col]}, Q3: {q3[col]}, IQR: {iqr[col]}')
    print(f'하한: {lower[col]}, 상한: {upper[col]}')
    rows = np.where(mask[:, col])[0]
    print('이상치 행 위치:', rows)
    print('이상치 값:', aaa[rows, col])

# 데이터의 모든 이상치 위치 (행, 열)
outlier_rows, outlier_cols = np.where(mask)
print('\n전체 이상치 위치 (행, 열):')
print(list(zip(outlier_rows, outlier_cols)))

# 열마다 스케일이 다르므로 각 열의 상자그림을 따로 그린다.
fig, axes = plt.subplots(1, aaa.shape[1], figsize=(10, 4))
for col, ax in enumerate(axes):
    ax.boxplot(aaa[:, col])
    ax.set_title(f'Column {col}')
    ax.set_ylabel('value')
plt.tight_layout()
plt.show()

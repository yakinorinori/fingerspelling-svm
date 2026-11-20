import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# CSVデータの読み込み
# df1 = pd.read_csv('Finger_data.csv')
# df2 = pd.read_csv('Finger_answer.csv')
df1 = pd.read_csv('Finger_data_move.csv')
df2 = pd.read_csv('Finger_answer_move.csv')

# DataFrameをNumpy配列に変換
X = df1.iloc[:, 1:]
Y = df2.iloc[:, 0]

# 各列を数値に変換し、欠損値をNaNに変換
X = X.apply(lambda col: pd.to_numeric(col, errors='coerce'))

# NaNを0に変換
X = np.nan_to_num(X)

# XとYの行数が一致することを確認
if len(X) != len(Y):
    min_len = min(len(X), len(Y))
    X = X[:min_len]
    Y = Y[:min_len]

# データを訓練データとテストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, shuffle=True, stratify=Y)


# クラスごとのデータ数に応じた重み（少ないクラスほど重い）
#   重み = 全データ数 / (クラス数 × そのクラスのデータ数)
classes, counts = np.unique(Y_train, return_counts=True)
print("クラスごとのデータ数と重み:")
for c, n in zip(classes, counts):
    print(f"  {c}: {n} 件, 重み {len(Y_train) / (len(classes) * n):.3f}")

# SVMモデルの訓練
svc = SVC(probability=True, class_weight="balanced")
svc.fit(X_train,Y_train)

# テストデータでの予測と評価
Y_test_pred = svc.predict(X_test)
accuracy = accuracy_score(Y_test,Y_test_pred)

print(f'Test Accuracy: {accuracy}')

# トレーニングデータの特徴の数を表示
print("Training data features:", X_train.shape[1])
# 学習したモデルの保存
# with open('hand.pickle', mode='wb') as f:
#     pickle.dump(svc, f, protocol=2)
# 学習したモデルの保存
with open('hand_move.pickle', mode='wb') as f:
    pickle.dump(svc, f, protocol=2)

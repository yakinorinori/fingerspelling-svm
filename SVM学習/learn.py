"""
特徴量 CSV から手型認識用の SVM を学習して保存する。

旧版からの変更点:
- 特徴量を StandardScaler で標準化してから SVM に入力する（比の特徴と角度の特徴で尺度が大きく違うため）。
  保存するのは「標準化 + SVM」をまとめた Pipeline で、predict / predict_proba / classes_ は
  これまでどおり使えるので、Adding_features.py などの呼び出し側は変更不要。
- 評価を交差検証にした（旧版は test_size=0.01 で、評価がほぼ意味をなしていなかった）。
- --groups で収録単位（表現者・動画など）のグループを与えると、グループ単位で分割して評価する。
  同じ動画の連続フレームが学習と評価の両方に入ると精度が過大になるため、論文の評価ではこちらを使う。
- --no-scale で旧版と同じ標準化なしの SVM を評価でき、標準化の効果を比較できる。

使い方:
    python learn.py --data Finger_data_move.csv --answer Finger_answer_move.csv --out hand_move.pickle
    python learn.py --data Finger_data.csv --answer Finger_answer.csv --groups Finger_group.csv --out hand.pickle
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_xy(data_path, answer_path):
    df1 = pd.read_csv(data_path)
    df2 = pd.read_csv(answer_path)

    # 旧版と同じく 1 列目は特徴量として使わない
    X = df1.iloc[:, 1:].apply(lambda col: pd.to_numeric(col, errors='coerce'))
    X = np.nan_to_num(X.to_numpy(dtype=float))
    Y = df2.iloc[:, 0].to_numpy()

    if len(X) != len(Y):
        raise ValueError(f'特徴量の行数 ({len(X)}) と正解の行数 ({len(Y)}) が一致しません。'
                         '行の対応がずれている可能性があるので、データを確認してください。')
    return X, Y


def build_model(scale):
    svc = SVC(probability=True)
    return make_pipeline(StandardScaler(), svc) if scale else svc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='Finger_data_move.csv')
    ap.add_argument('--answer', default='Finger_answer_move.csv')
    ap.add_argument('--groups', help='各行のグループ（表現者・動画 ID など）を 1 列目に持つ CSV')
    ap.add_argument('--out', default='hand_move.pickle')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--no-scale', action='store_true', help='標準化しない（旧版と同じ）')
    args = ap.parse_args()

    X, Y = load_xy(args.data, args.answer)
    print(f'サンプル数: {len(X)}, 特徴量の数: {X.shape[1]}, クラス数: {len(np.unique(Y))}')

    model = build_model(scale=not args.no_scale)

    if args.groups:
        groups = pd.read_csv(args.groups).iloc[:, 0].to_numpy()
        if len(groups) != len(X):
            raise ValueError('グループの行数が特徴量の行数と一致しません。')
        n_groups = len(np.unique(groups))
        folds = min(args.folds, n_groups)
        cv = GroupKFold(n_splits=folds)
        scores = cross_val_score(model, X, Y, cv=cv, groups=groups)
        print(f'グループ単位 {folds} 分割交差検証（{n_groups} グループ）')
    else:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
        scores = cross_val_score(model, X, Y, cv=cv)
        print(f'層化 {args.folds} 分割交差検証（注意: 同じ動画の連続フレームが学習と評価の両方に入るため、'
              '精度は過大に出る。論文の評価では --groups を使うこと）')
    print(f'Accuracy: {scores.mean():.4f} ± {scores.std():.4f}  (各分割: {np.round(scores, 4).tolist()})')

    # 全データで学習して保存
    model.fit(X, Y)
    with open(args.out, mode='wb') as f:
        pickle.dump(model, f, protocol=2)
    print(f'モデルを保存しました: {args.out}')


if __name__ == '__main__':
    main()

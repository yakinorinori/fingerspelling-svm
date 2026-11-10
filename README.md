# fingerspelling-svm

SVM による静止指文字の学習と、クラス類似度分析。

## 構成（SVM学習/）

- `learn.py` — 特徴量 CSV から SVM（SVC）を学習しモデルを保存
- `SVMlearn.py` — （作業用・空ファイル）
- `SVMSimilarityAnalyzer.py` — 学習済み SVM のクラス間類似度分析・階層クラスタリング・デンドログラム生成
- `generate_10class_structure.py` — 類似度分析に基づき 5 クラス構造を 10 クラスに細分化
- `skeleton_10class_config.py` — 10 クラス構造の文字→クラス対応表
- `skeleton_10class_mapping.json` — 10 クラス構造のマッピング定義

## 必要ライブラリ
`scikit-learn`, `numpy`, `pandas`, `matplotlib`

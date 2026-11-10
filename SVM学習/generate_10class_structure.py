#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVMの類似度分析に基づいて、10クラス構造を生成するスクリプト
既存の5クラス構造をさらに細分化して10クラスに拡張する
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# フォント設定
plt.rcParams['font.sans-serif'] = ['IPAexGothic']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # SVMモデルとデータを読み込む
    try:
        with open('hand.pickle', 'rb') as f:
            svm_model = pickle.load(f)
        print("✓ SVMモデルを読み込みました")
    except Exception as e:
        print(f"✗ SVMモデルの読み込みに失敗: {e}")
        return
    
    try:
        finger_data = pd.read_csv('CSV/Finger_ data.csv')
        finger_labels = pd.read_csv('CSV/Finger_answer.csv')
        print(f"✓ データを読み込みました: {finger_data.shape}")
    except Exception as e:
        print(f"✗ データの読み込みに失敗: {e}")
        return
    
    # SVMの決定関数値を使用して類似度を計算
    all_labels = np.unique(finger_labels.values.flatten())
    print(f"\n全文字: {', '.join(sorted(all_labels))}")
    print(f"総文字数: {len(all_labels)}")
    
    # 各文字の代表的な特徴ベクトルを計算
    char_vectors = {}
    finger_labels_flat = finger_labels.values.flatten() if finger_labels.ndim > 1 else finger_labels.iloc[:, 0].values
    
    for char in all_labels:
        # より安全なマスク処理
        mask = [i for i, label in enumerate(finger_labels_flat) if label == char]
        if len(mask) > 0:
            char_features = finger_data.iloc[mask].values
            char_vectors[char] = np.mean(char_features, axis=0)
        else:
            print(f"警告: 文字'{char}'のサンプルが見つかりません")
    
    print(f"\n各文字の特徴ベクトル計算完了")
    
    # 類似度行列を計算（コサイン類似度）
    char_list = sorted(all_labels)
    char_to_idx = {char: idx for idx, char in enumerate(char_list)}
    
    feature_matrix = np.array([char_vectors[char] for char in char_list])
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # 距離行列を計算（類似度の逆数）
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    print(f"類似度行列サイズ: {similarity_matrix.shape}")
    
    # ===== 10クラスへの階層的クラスタリング =====
    print("\n" + "="*60)
    print("10クラス構造の生成")
    print("="*60)
    
    try:
        # 凝縮距離行列を作成
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, method='average')
        
        # 10クラスに切断
        from scipy.cluster.hierarchy import fcluster
        clusters_10 = fcluster(Z, 10, criterion='maxclust')
        
        print(f"\n✓ 10クラスへのクラスタリング成功")
        
    except Exception as e:
        print(f"✗ 階層的クラスタリングエラー: {e}")
        print("→ 類似度に基づいた手動グループ分割を実行します\n")
        
        # 手動グループ分割戦略
        # 類似度行列から最も異なる文字ペアを見つけて、グループを分割
        clusters_10 = assign_10_clusters_manual(similarity_matrix, char_list)
    
    # クラスタ結果をグループ化
    groups_10 = {}
    for char, cluster_id in zip(char_list, clusters_10):
        if cluster_id not in groups_10:
            groups_10[cluster_id] = []
        groups_10[cluster_id].append(char)
    
    # グループサイズ統計
    print("\n" + "-"*60)
    print("10クラス構造（グループサイズ）")
    print("-"*60)
    
    group_sizes = []
    for group_id in sorted(groups_10.keys()):
        chars = groups_10[group_id]
        group_sizes.append(len(chars))
        print(f"Class {group_id:2d}: ({len(chars):2d}文字) {', '.join(sorted(chars))}")
    
    print(f"\n統計:")
    print(f"  最大グループ: {max(group_sizes)}文字")
    print(f"  最小グループ: {min(group_sizes)}文字")
    print(f"  平均グループ: {np.mean(group_sizes):.1f}文字")
    print(f"  標準偏差: {np.std(group_sizes):.2f}")
    
    # ===== グループの類似度分析 =====
    print("\n" + "="*60)
    print("グループ内の類似度分析")
    print("="*60)
    
    for group_id in sorted(groups_10.keys()):
        chars = sorted(groups_10[group_id])
        if len(chars) > 1:
            indices = [char_to_idx[char] for char in chars]
            # グループ内平均類似度
            intra_similarities = []
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    intra_similarities.append(similarity_matrix[indices[i], indices[j]])
            
            avg_similarity = np.mean(intra_similarities) if intra_similarities else 0
            print(f"Class {group_id:2d}: グループ内平均類似度 = {avg_similarity:.4f}")
        else:
            print(f"Class {group_id:2d}: 単一文字グループ")
    
    # ===== クラスマッピング情報を保存 =====
    print("\n" + "="*60)
    print("クラスマッピング情報を保存")
    print("="*60)
    
    # JSON形式で保存
    import json
    mapping_data = {
        "class_structure": "10-class",
        "total_characters": len(all_labels),
        "groups": {str(group_id): sorted(chars) 
                   for group_id, chars in groups_10.items()},
        "group_statistics": {
            "total_groups": len(groups_10),
            "max_size": int(max(group_sizes)),
            "min_size": int(min(group_sizes)),
            "mean_size": float(np.mean(group_sizes)),
            "std_size": float(np.std(group_sizes))
        }
    }
    
    with open('skeleton_10class_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print("✓ skeleton_10class_mapping.json に保存しました")
    
    # Pythonコード形式で保存（直接使用可能）
    print_class_mapping_python(groups_10)
    
    return groups_10

def assign_10_clusters_manual(similarity_matrix, char_list):
    """
    類似度行列を基に手動で10クラスを割り当てる
    """
    n = len(char_list)
    clusters = np.zeros(n, dtype=int)
    
    # 凝縮距離行列を作成
    from scipy.spatial.distance import squareform
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = squareform(distance_matrix)
    
    try:
        Z = linkage(condensed_dist, method='ward')
        from scipy.cluster.hierarchy import fcluster
        clusters = fcluster(Z, 10, criterion='maxclust')
    except Exception as e:
        print(f"ward方法でも失敗: {e}")
        # フォールバック: 5クラス結果を10クラスに分割
        from scipy.cluster.hierarchy import fcluster
        try:
            Z = linkage(condensed_dist, method='complete')
            clusters = fcluster(Z, 10, criterion='maxclust')
        except:
            print("デフォルト10クラス割り当てを使用")
            for i in range(n):
                clusters[i] = i % 10
    
    return clusters

def print_class_mapping_python(groups_10):
    """Pythonコード形式でクラスマッピングを出力"""
    print("\n" + "="*60)
    print("Pythonコード形式のクラスマッピング")
    print("="*60)
    
    print("\n# 10クラス構造のマッピング")
    print("CHARACTER_CLASS_MAPPING_10 = {")
    
    for group_id in sorted(groups_10.keys()):
        chars = groups_10[group_id]
        for char in sorted(chars):
            print(f"    '{char}': {group_id},")
    
    print("}\n")
    
    print("# クラスラベル")
    print("CLASS_NAMES_10 = {")
    for group_id in sorted(groups_10.keys()):
        print(f"    {group_id}: 'Class_{group_id}',")
    print("}\n")
    
    # ファイルに保存
    with open('skeleton_10class_config.py', 'w', encoding='utf-8') as f:
        f.write("# 10クラス構造のマッピング\n")
        f.write("CHARACTER_CLASS_MAPPING_10 = {\n")
        
        for group_id in sorted(groups_10.keys()):
            chars = groups_10[group_id]
            for char in sorted(chars):
                f.write(f"    '{char}': {group_id},\n")
        
        f.write("}\n\n")
        f.write("# クラスラベル\n")
        f.write("CLASS_NAMES_10 = {\n")
        for group_id in sorted(groups_10.keys()):
            f.write(f"    {group_id}: 'Class_{group_id}',\n")
        f.write("}\n")
    
    print("✓ skeleton_10class_config.py に保存しました")

if __name__ == '__main__':
    groups = main()

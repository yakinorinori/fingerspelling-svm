import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# macOS用の日本語フォント設定
plt.rcParams['font.family'] = 'IPAexGothic'  
plt.rcParams['axes.unicode_minus'] = False

class PreTrainedModelLoader:
    """PyTorchとPickleモデルの統一的なローダー"""
    
    @staticmethod
    def load_model(model_path):
        """モデルを自動判別して読み込む"""
        if model_path.endswith('.pth'):
            # PyTorch モデルの読み込み（.pthファイルが見つかった場合のメッセージ）
            print(f"警告: PyTorch モデル ({model_path}) が見つかりました。")
            print("このプログラムはPickleモデルに対応しています。")
            print("以下の pickle モデルを使用してください:")
            print("  - hand.pickle")
            print("  - finalized_model.sav")
            raise ValueError(f"PyTorch モデルはサポートしていません: {model_path}")
        elif model_path.endswith('.pickle') or model_path.endswith('.pkl') or model_path.endswith('.sav'):
            # Pickle モデルの読み込み
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Pickle モデルを読み込みました: {model_path}")
            print(f"モデルの型: {type(model)}")
            return model, 'pickle'
        else:
            raise ValueError(f"不支持なモデル形式: {model_path}")

class PreTrainedSVMSimilarityMapper:
    def __init__(self, 
                data_path, 
                answer_path, 
                model_path='hand.pickle'):
        # モデルの読み込み
        try:
            self.model, self.model_type = PreTrainedModelLoader.load_model(model_path)
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            raise

        # データと正解ラベルの読み込み
        self.X, self.y = self._load_data(data_path, answer_path)
        
        # データの標準化
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        self.model_type_used = self.model_type

    def _load_data(self, data_path, answer_path):
        try:
            # データファイル読み込み
            data_df = pd.read_csv(data_path, header=0)
            
            # 一番左の列を削除
            X = data_df.iloc[:, 1:].values
            
            # ラベルファイル読み込み
            labels_df = pd.read_csv(answer_path, header=0)
            y = labels_df.iloc[:, 0].values
            
            print("データ情報:")
            print(f"特徴量形状: {X.shape}")
            print(f"ラベル数: {len(y)}")
            print(f"ユニークラベル: {np.unique(y)}")
            
            return X, y
        
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None, None

    def compute_similarity(self):
        try:
            # 予測確率や決定関数を使用した類似度計算
            unique_classes = np.unique(self.y)
            
            # クラスごとの代表的な特徴ベクトルを計算
            class_representatives = {}
            for cls in unique_classes:
                # そのクラスのデータポイント
                class_mask = (self.y == cls)
                class_data = self.X_scaled[class_mask]
                
                # クラスの代表ベクトル（平均）
                class_representatives[cls] = np.mean(class_data, axis=0)
            
            # クラス間の類似度計算
            class_list = list(class_representatives.keys())
            n_classes = len(class_list)
            similarity_matrix = np.zeros((n_classes, n_classes))
            
            for i in range(n_classes):
                for j in range(n_classes):
                    # クラス代表ベクトル間の類似度計算
                    vec_i = class_representatives[class_list[i]]
                    vec_j = class_representatives[class_list[j]]
                    
                    # コサイン類似度
                    cos_sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                    
                    # ユークリッド距離ベースの類似度
                    euclidean_dist = np.linalg.norm(vec_i - vec_j)
                    dist_sim = 1 / (1 + euclidean_dist)
                    
                    # 複合類似度
                    similarity = (cos_sim + dist_sim) / 2
                    
                    similarity_matrix[i, j] = similarity
            
            # 視覚化
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix, 
                        xticklabels=class_list, 
                        yticklabels=class_list, 
                        annot=True, 
                        cmap='YlGnBu', 
                        fmt='.4f')
            plt.title('クラス間類似度マトリクス')
            plt.tight_layout()
            plt.savefig('class_similarity_matrix.png')
            plt.close()
            
            # クラスの平均類似度に基づいてランキング
            avg_similarities = np.mean(similarity_matrix, axis=1)
            ranking = np.argsort(avg_similarities)[::-1]  # 降順でソート
            
            # ランキングに基づいて辞書を作成
            class_ranking = {
                class_list[idx]: rank + 1 
                for rank, idx in enumerate(ranking)
            }
            
            return class_ranking, similarity_matrix, class_list
        
        except Exception as e:
            print(f"類似度計算中のエラー: {e}")
            return {}, None, None

def group_similar_signs(similarity_matrix, class_list, n_clusters=5):
    """
    類似度行列を用いて手話文字をクラスタリング
    
    Parameters:
    - similarity_matrix: クラス間の類似度行列
    - class_list: クラスのリスト
    - n_clusters: クラスターの数
    
    Returns:
    - クラスター分けされた手話文字グループ
    """
    # 距離行列に変換 (1 - 類似度)
    distance_matrix = 1 - similarity_matrix
    
    # 階層的クラスタリング
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        linkage='average'
    )
    
    # クラスタリングの実行
    clusters = clustering.fit_predict(distance_matrix)
    
    # クラスターごとにグループ化
    clustered_groups = {}
    for i in range(n_clusters):
        clustered_groups[i] = [
            class_list[j] for j in range(len(class_list)) if clusters[j] == i
        ]
    
    return clustered_groups, clusters

def visualize_clusters(similarity_matrix, class_list, clusters):
    """
    クラスター分析の結果をヒートマップで可視化
    """
    # 並べ替えのためのインデックス
    sorted_indices = np.argsort(clusters)
    
    # 並べ替えた類似度行列
    sorted_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
    sorted_classes = [class_list[i] for i in sorted_indices]
    
    # 可視化
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sorted_matrix, 
        xticklabels=sorted_classes, 
        yticklabels=sorted_classes, 
        annot=True, 
        cmap='YlGnBu', 
        fmt='.4f'
    )
    plt.title(f'クラスター別の類似度マトリクス')
    plt.tight_layout()
    plt.savefig('clustered_similarity_matrix.png')
    plt.close()

def create_dendrogram(similarity_matrix, class_list):
    """
    階層的クラスタリングの結果をデンドログラムで可視化
    """
    # 距離行列に変換
    distance_matrix = 1 - similarity_matrix
    
    # リンケージ行列を計算
    Z = linkage(distance_matrix, method='average')
    
    # デンドログラムを描画
    plt.figure(figsize=(16, 8))
    dendrogram(
        Z,
        labels=class_list,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_font_size=12
    )
    plt.title('手話文字の類似度に基づく階層的クラスタリング')
    plt.xlabel('手話文字')
    plt.ylabel('距離')
    plt.tight_layout()
    plt.savefig('sign_language_dendrogram.png')
    plt.close()

def find_optimal_clusters(similarity_matrix, max_clusters=10):
    """
    シルエットスコアを用いて最適なクラスター数を見つける
    """
    # 距離行列に変換
    distance_matrix = 1 - similarity_matrix
    
    # 対角線を0に設定
    np.fill_diagonal(distance_matrix, 0)
    
    # condensed距離行列に変換
    condensed_distance = squareform(distance_matrix)
    
    # 可能なクラスター数のリスト
    cluster_range = range(2, min(max_clusters + 1, len(distance_matrix)))
    
    # 各クラスター数のシルエットスコアを計算
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        try:
            # 階層的クラスタリングのリンケージ計算
            Z = linkage(condensed_distance, method='average')
            
            # クラスターラベルの生成
            clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
            clusters = clusters - 1  # 0から始まるインデックスに調整
            
            # 一意のクラスターが存在することを確認
            if len(np.unique(clusters)) < 2:
                continue
            
            # シルエットスコアを計算
            score = silhouette_score(distance_matrix, clusters, metric='precomputed')
            silhouette_scores.append(score)
        except Exception as e:
            print(f"クラスター数 {n_clusters} の評価中にエラー: {e}")
            silhouette_scores.append(-1)
    
    # 最適なクラスター数を選択
    if silhouette_scores and max(silhouette_scores) > -1:
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    else:
        print("シルエットスコアの計算に失敗しました。デフォルト値を使用します。")
        optimal_clusters = 3
    
    # 結果の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('クラスター数とシルエットスコア')
    plt.xlabel('クラスター数')
    plt.ylabel('シルエットスコア')
    plt.tight_layout()
    plt.savefig('silhouette_scores.png')
    plt.close()
    
    print(f"\n最適なクラスター数: {optimal_clusters}")
    print("シルエットスコアのグラフを保存しました: silhouette_scores.png")
    
    return optimal_clusters

def flexible_clustering(similarity_matrix, class_list, clustering_method='auto', n_clusters=None):
    """
    柔軟なクラスタリング方法
    """
    # 距離行列に変換
    distance_matrix = 1 - similarity_matrix
    
    # 対角線を0に設定
    np.fill_diagonal(distance_matrix, 0)
    
    # クラスター数の決定
    if clustering_method == 'auto':
        try:
            n_clusters = find_optimal_clusters(similarity_matrix)
        except Exception as e:
            print(f"最適クラスター数の自動決定に失敗: {e}")
            n_clusters = min(3, len(class_list) - 1)  # クラスター数を制限
    elif clustering_method == 'manual':
        if n_clusters is None:
            n_clusters = min(3, len(class_list) - 1)
    
    # クラスタリング
    try:
        # condensed距離行列に変換
        condensed_distance = squareform(distance_matrix)
        
        # 階層的クラスタリングのリンケージ計算
        Z = linkage(condensed_distance, method='average')
        
        # クラスターラベルの生成
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
        clusters = clusters - 1  # 0から始まるインデックスに調整
    except Exception as e:
        print(f"クラスタリング中のエラー: {e}")
        # フォールバック：単純な均等分割
        clusters = np.array([i % n_clusters for i in range(len(class_list))])
    
    # クラスターごとにグループ化
    clustered_groups = {}
    for i in range(n_clusters):
        clustered_groups[i] = [
            class_list[j] for j in range(len(class_list)) if clusters[j] == i
        ]
    
    return clustered_groups, clusters

def set_japanese_font():
    """
    日本語フォントを設定する関数
    """
    import matplotlib
    
    # macOS用のフォントリスト
    font_candidates = [
        'IPAexGothic',
        'Hiragino Sans',
        'Yu Gothic',
        'Meiryo'
    ]
    
    for font in font_candidates:
        try:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['figure.figsize'] = (10, 8)  # デフォルトのグラフサイズ
            print(f"フォント設定: {font}")
            return
        except Exception:
            continue
    
    print("適切な日本語フォントが見つかりませんでした。")

def main():
    # データと正解ラベルのパス
    data_path = 'Finger_data.csv'
    answer_path = 'Finger_answer.csv'
    
    # 学習済みモデルのパス（複数の候補をサポート）
    model_candidates = [
        'hand.pickle',  # SVM モデル
        'finalized_model.sav',  # 別のSVMモデル
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break
    
    if model_path is None:
        print(f"利用可能なモデルが見つかりません。")
        print(f"検索対象: {model_candidates}")
        return
    
    print(f"使用するモデル: {model_path}\n")
    set_japanese_font()
    
    # 学習済みモデルを使用した類似度マッパーを初期化
    try:
        svm_mapper = PreTrainedSVMSimilarityMapper(
            data_path, 
            answer_path, 
            model_path
        )
    except Exception as e:
        print(f"マッパーの初期化に失敗: {e}")
        return
    
    # 類似度計算
    try:
        class_ranking, similarity_matrix, class_list = svm_mapper.compute_similarity()
    except Exception as e:
        print(f"類似度計算中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 結果表示
    print("\n文字から数値へのマッピング:")
    for char, rank in sorted(class_ranking.items(), key=lambda x: x[1]):
        print(f"{char}: {rank}")
    
    # クラスタリング方法の比較
    clustering_configs = [
        {'method': 'auto', 'n_clusters': None, 'key': 'auto'},
        {'method': 'manual', 'n_clusters': 3, 'key': '3_clusters'},
        {'method': 'manual', 'n_clusters': 4, 'key': '4_clusters'},
        {'method': 'manual', 'n_clusters': 5, 'key': '5_clusters'}
    ]
    
    results = {}
    for config in clustering_configs:
        try:
            similar_groups, clusters = flexible_clustering(
                similarity_matrix, 
                class_list, 
                clustering_method=config['method'], 
                n_clusters=config['n_clusters']
            )
            
            # 結果の保存とログ出力
            results[config['key']] = {
                'groups': similar_groups,
                'clusters': clusters
            }
            
            # クラスタリング結果の表示
            print(f"\n{config['key']} クラスタリングのグループ:")
            for group_num, group_chars in similar_groups.items():
                print(f"グループ {group_num + 1}: {', '.join(group_chars)}")
        
        except Exception as e:
            print(f"クラスタリング設定 {config} でエラー: {e}")
    
    # 可視化
    # 自動クラスタリングが失敗した場合は最初の手動クラスタリング結果を使用
    default_clusters = results.get('auto', results.get('3_clusters', {}).get('clusters'))
    
    if default_clusters is not None:
        visualize_clusters(similarity_matrix, class_list, default_clusters)
        create_dendrogram(similarity_matrix, class_list)
        print("\n可視化ファイルを生成しました:")
        print("  - clustered_similarity_matrix.png")
        print("  - sign_language_dendrogram.png")
    else:
        print("クラスタリングの可視化に失敗しました。")
    
    print("\n" + "="*70)
    print("分析完了")
    print("="*70)
    
    return results

# 可視化関数の修正
def visualize_clusters(similarity_matrix, class_list, clusters):
    """
    クラスター分析の結果をヒートマップで可視化
    """
    # 並べ替えのためのインデックス
    sorted_indices = np.argsort(clusters)
    
    # 並べ替えた類似度行列
    sorted_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
    sorted_classes = [class_list[i] for i in sorted_indices]
    
    # 可視化
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sorted_matrix, 
        xticklabels=sorted_classes, 
        yticklabels=sorted_classes, 
        annot=True, 
        cmap='YlGnBu', 
        fmt='.4f'
    )
    plt.title(f'クラスター別の類似度マトリクス')
    plt.tight_layout()
    plt.savefig('clustered_similarity_matrix.png')
    plt.close()

# メイン関数の呼び出し
if __name__ == "__main__":
    main()
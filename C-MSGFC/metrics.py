"""from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import accuracy_score
import numpy as np
from munkres import Munkres

def best_map(L1, L2):
    Label1 = np.unique(L1)
    n_class1 = len(Label1)
    Label2 = np.unique(L2)
    n_class2 = len(Label2)
    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))
    for i in range(len(L1)):
        # 确保标签在G的索引范围内
        if L1[i] < n_class and L2[i] < n_class:
            G[L1[i], L2[i]] += 1
    m = Munkres()
    indexes = m.compute(G.max() - G)
    new_L2 = np.zeros(len(L2), dtype=int)
    for i in range(len(indexes)):
        # 确保索引在原始标签中存在
        true_label_val = indexes[i][0]
        pred_label_val = indexes[i][1]
        new_L2[L2 == pred_label_val] = true_label_val
    return new_L2

def cal_clustering_metric(truth, prediction):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)

    # NMI 和 ARI 的计算是正确的，因为它们与标签值无关
    nmi = normalized_mutual_info_score(truth, prediction)
    ari = adjusted_rand_score(truth, prediction)

    # ACC 和 F1-Score 需要先进行标签匹配
    # 使用 try-except 以处理可能的匹配失败（例如，当预测的簇数与真实类别数差别很大时）
    try:
        y_pred_mapped = best_map(truth, prediction)
        acc = accuracy_score(truth, y_pred_mapped)
        # 【核心修正】使用匹配后的标签计算 F1-Score
        f1 = f1_score(truth, y_pred_mapped, average='macro')
    except Exception as e:
        print(f"Warning: Failed to compute mapped metrics (ACC, F1). Error: {e}")
        acc = 0.0
        f1 = 0.0

    return acc, nmi, ari, f1"""
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import accuracy_score
import numpy as np
from munkres import Munkres


def best_map(L1, L2):
    """
    使用匈牙利算法（Munkres）找到 L2 到 L1 的最佳标签映射。
    此版本经过修正，可以处理任意不连续的整数标签。
    """
    L1 = np.asarray(L1)
    L2 = np.asarray(L2)

    # --- 核心修正：将原始标签映射到连续的0-N范围 ---
    Label1 = np.unique(L1)
    n_class1 = len(Label1)
    map1 = {label: i for i, label in enumerate(Label1)}
    L1_mapped = np.array([map1[label] for label in L1])

    Label2 = np.unique(L2)
    n_class2 = len(Label2)
    map2 = {label: i for i, label in enumerate(Label2)}
    L2_mapped = np.array([map2[label] for label in L2])
    # --- 修正结束 ---

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    # 使用映射后的、安全的标签来构建匹配矩阵
    for i in range(len(L1_mapped)):
        G[L1_mapped[i], L2_mapped[i]] += 1

    m = Munkres()
    cost_matrix = G.max() - G
    indexes = m.compute(cost_matrix)

    # 创建一个从旧的预测标签到新的真实标签的映射字典
    pred_to_true_map = {}
    for true_idx, pred_idx in indexes:
        # 找到原始的真实标签和预测标签
        # true_idx 对应于 Label1 的索引，pred_idx 对应于 Label2 的索引
        if true_idx < n_class1 and pred_idx < n_class2:
            original_true_label = Label1[true_idx]
            original_pred_label = Label2[pred_idx]
            pred_to_true_map[original_pred_label] = original_true_label

    # 使用这个映射来生成最终的、重新标记的预测数组
    new_L2 = np.array([pred_to_true_map.get(l, l) for l in L2])
    # .get(l, l) 表示如果某个预测标签不在映射中（簇数不匹配时可能发生），则保持原样

    return new_L2


# cal_clustering_metric 函数保持不变，因为它已经是正确的了
def cal_clustering_metric(truth, prediction):
    # ... (您的 cal_clustering_metric 函数代码可以完全保持原样)
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    nmi = normalized_mutual_info_score(truth, prediction)
    ari = adjusted_rand_score(truth, prediction)
    try:
        y_pred_mapped = best_map(truth, prediction)
        acc = accuracy_score(truth, y_pred_mapped)
        f1 = f1_score(truth, y_pred_mapped, average='macro')
    except Exception as e:
        print(f"Warning: Failed to compute mapped metrics (ACC, F1). Error: {e}")
        acc = 0.0
        f1 = 0.0
    return acc, nmi, ari, f1
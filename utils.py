import numpy as np
import scipy.sparse as sp
import scipy.io as scio
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph

def build_knn_graph(X, k=10):
    """
    基于特征矩阵构建 KNN 邻接图（无权、稀疏）
    """
    print(f"📐 Building KNN graph (k={k}) ...")
    A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True)
    A = A + A.T  # 对称化
    A[A > 1] = 1
    A = sp.coo_matrix(A)
    print(f"✅ KNN graph built: shape={A.shape}, edges={A.nnz}")
    return A

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    return np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

def normalize_features(mx):
    row_sum = np.array(mx.sum(1))
    row_inv = np.power(row_sum, -1, where=row_sum != 0).flatten()
    row_inv[np.isinf(row_inv)] = 0.
    row_mat_inv = sp.diags(row_inv)
    return row_mat_inv.dot(mx)

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def remove_isolated_nodes(features, adj, labels):
    adj_dense = adj.toarray() if hasattr(adj, 'toarray') else adj
    num_components, component_labels = csgraph.connected_components(adj_dense, directed=False, return_labels=True)
    main_component = (component_labels == np.bincount(component_labels).argmax())
    return features[main_component], adj[main_component][:, main_component], labels[main_component]

def load_citeseer():
    path = 'data/citeseer/'
    data_name = 'citeseer'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt(f"{path}{data_name}.content", dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = normalize_features(features)
    labels = idx_features_labels[:, -1]
    _, _, labels = np.unique(labels, return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=str)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt(f"{path}{data_name}.cites", dtype=np.dtype(str))
    edge_list = []
    for src, dst in edges_unordered:
        if src in idx_map and dst in idx_map:
            edge_list.append([idx_map[src], idx_map[dst]])
    edges = np.array(edge_list, dtype=np.int32)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    features, adj, labels = remove_isolated_nodes(features, adj, labels)
    return features.todense(), idx_map, adj.toarray(), labels

def load_cora():
    path = 'data/cora/'
    data_name = 'cora'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features.toarray(), idx_map, adj.toarray(), labels

def load_wiki():
    print("📥 Loading WIKI dataset from .npy files...")

    features = np.load('data/wiki/wiki_feat.npy', allow_pickle=True)
    adj = np.load('data/wiki/wiki_adj.npy', allow_pickle=True)
    labels = np.load('data/wiki/wiki_label.npy', allow_pickle=True)

    # ✅ 特征类型标准化
    if features.dtype == np.object_:
        features = np.array(features.tolist(), dtype=np.float32)
    else:
        features = features.astype(np.float32)

    # ✅ 转为稀疏格式
    features = sp.csr_matrix(features)
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)

    # ✅ 特征归一化 + 邻接加自环 + 归一化
    features = normalize_features(features)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)

    print(f"✅ Loaded WIKI: features={features.shape}, adj={adj.shape}, labels={np.unique(labels).size} classes")
    return features.toarray(), None, adj.toarray(), labels

def load_amap():
    print("📥 Loading AMAP dataset from .npy files...")

    features = np.load('data/amap/amap_feat.npy', allow_pickle=True)
    adj = np.load('data/amap/amap_adj.npy', allow_pickle=True)
    labels = np.load('data/amap/amap_label.npy', allow_pickle=True)

    # ✅ 特征类型标准化
    if features.dtype == np.object_:
        features = np.array(features.tolist(), dtype=np.float32)
    else:
        features = features.astype(np.float32)

    # ✅ 转为稀疏格式
    features = sp.csr_matrix(features)
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)

    # ✅ 特征归一化 + 邻接加自环 + 归一化
    features = normalize_features(features)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)

    print(f"✅ Loaded AMAP: features={features.shape}, adj={adj.shape}, labels={np.unique(labels).size} classes")
    return features.toarray(), None, adj.toarray(), labels
def load_acm():
    print("📥 Loading ACM dataset from .npy files...")

    features = np.load('data/acm/acm_feat.npy', allow_pickle=True)
    adj = np.load('data/acm/acm_adj.npy', allow_pickle=True)
    labels = np.load('data/acm/acm_label.npy', allow_pickle=True)

    # ✅ 特征类型标准化
    if features.dtype == np.object_:
        features = np.array(features.tolist(), dtype=np.float32)
    else:
        features = features.astype(np.float32)

    # ✅ 转为稀疏格式
    features = sp.csr_matrix(features)
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)

    # ✅ 特征归一化 + 邻接加自环 + 归一化
    features = normalize_features(features)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)

    print(f"✅ Loaded ACM: features={features.shape}, adj={adj.shape}, labels={np.unique(labels).size} classes")
    return features.toarray(), None, adj.toarray(), labels

def load_dblp():
    print("📥 Loading DBLP dataset from .npy files...")

    features = np.load('data/dblp/dblp_feat.npy', allow_pickle=True)
    adj = np.load('data/dblp/dblp_adj.npy', allow_pickle=True)
    labels = np.load('data/dblp/dblp_label.npy', allow_pickle=True)

    # ✅ 特征类型标准化
    if features.dtype == np.object_:
        features = np.array(features.tolist(), dtype=np.float32)
    else:
        features = features.astype(np.float32)

    # ✅ 转为稀疏格式
    features = sp.csr_matrix(features)
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)

    # ✅ 特征归一化 + 邻接加自环 + 归一化
    features = normalize_features(features)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)

    print(f"✅ Loaded DBLP: features={features.shape}, adj={adj.shape}, labels={np.unique(labels).size} classes")
    return features.toarray(), None, adj.toarray(), labels
def load_pubmed():
    print('Loading from raw data file...')
    data = scio.loadmat('data/pubmed.mat')
    adj = data['W']
    features = data['fea']
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    return features, None, adj.tocoo(), labels

def load_usps():
    print("📥 Loading USPS dataset from .npy files...")

    features = np.load('data/usps/usps_feat.npy', allow_pickle=True)
    labels = np.load('data/usps/usps_label.npy', allow_pickle=True)

    if features.dtype == np.object_:
        features = np.array(features.tolist(), dtype=np.float32)
    else:
        features = features.astype(np.float32)

    features = sp.csr_matrix(features)
    features = normalize_features(features)

    # ✅ 构建邻接矩阵
    adj = build_knn_graph(features.toarray(), k=15)
    adj = adj + sp.eye(adj.shape[0])  # 加自环
    adj = normalize_adj(adj)

    print(f"✅ Loaded USPS: features={features.shape}, labels={np.unique(labels).size} classes")
    return features.toarray(), adj.toarray(), labels

def load_hhar():
    print("📥 Loading HHAR dataset from .npy files...")

    features = np.load('data/hhar/hhar_feat.npy', allow_pickle=True)
    labels = np.load('data/hhar/hhar_label.npy', allow_pickle=True)

    if features.dtype == np.object_:
        features = np.array(features.tolist(), dtype=np.float32)
    else:
        features = features.astype(np.float32)

    features = sp.csr_matrix(features)
    features = normalize_features(features)

    # ✅ 构建邻接矩阵
    adj = build_knn_graph(features.toarray(), k=15)
    adj = adj + sp.eye(adj.shape[0])  # 加自环
    adj = normalize_adj(adj)

    print(f"✅ Loaded HHAR: features={features.shape}, labels={np.unique(labels).size} classes")
    return features.toarray(), adj.toarray(), labels

def load_reut():
    print("📥 Loading REUT dataset from .npy files...")

    features = np.load('data/reut/reut_feat.npy', allow_pickle=True)
    labels = np.load('data/reut/reut_label.npy', allow_pickle=True)

    if features.dtype == np.object_:
        features = np.array(features.tolist(), dtype=np.float32)
    else:
        features = features.astype(np.float32)

    features = sp.csr_matrix(features)
    features = normalize_features(features)

    # ✅ 构建邻接矩阵
    adj = build_knn_graph(features.toarray(), k=25)
    adj = adj + sp.eye(adj.shape[0])  # 加自环
    adj = normalize_adj(adj)

    print(f"✅ Loaded REUT: features={features.shape}, labels={np.unique(labels).size} classes")
    return features.toarray(), adj.toarray(), labels

def load_data(name):
    name = name.lower()
    if name == 'cora':
        features, _, adj, labels = load_cora()
        return features, adj, labels
    elif name == 'citeseer':
        features, _, adj, labels = load_citeseer()
        return features, adj, labels
    elif name == 'wiki':
        features, _, adj, labels = load_wiki()
        return features, adj, labels
    elif name == 'amap':
        features, _, adj, labels = load_amap()
        return features, adj, labels
    elif name == 'acm':
        features, _, adj, labels = load_acm()
        return features, adj, labels
    elif name == 'dblp':
        features, _, adj, labels = load_dblp()
        return features, adj, labels
    elif name == 'pubmed':
        features, _, adj, labels = load_pubmed()
        return features, adj, labels
    elif name == 'usps':
        features, adj, labels = load_usps()
        return features, adj, labels
    elif name == 'hhar':
        features, adj, labels = load_hhar()
        return features, adj, labels
    elif name == 'reut':
        features, adj, labels = load_reut()
        return features, adj, labels


    else:
        # 默认加载 .mat 格式数据
        path = 'data/{}.mat'.format(name)
        print(f"📥 Loading from {path} ...")
        data = scio.loadmat(path)
        labels = data['Y']
        labels = np.reshape(labels, (labels.shape[0],))
        adj = data['W']
        return data['X'], adj, labels


if __name__ == '__main__':
    load_data('YALE')

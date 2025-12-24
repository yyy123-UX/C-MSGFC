import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


# ----------------- 度量与工具 -----------------

def cal_clustering_metric(true_labels, pred_labels):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    ari = adjusted_rand_score(true_labels, pred_labels)
    acc, _, _, f1_aligned = cluster_acc_and_f1(true_labels, pred_labels)
    return acc, nmi, ari, f1_aligned


def cluster_acc_and_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-w)

    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size
    new_labels_map = {row: col for row, col in zip(row_ind, col_ind)}
    pred_labels_mapped = np.array([new_labels_map.get(l, -1) for l in y_pred])
    f1 = f1_score(y_true, pred_labels_mapped, average='macro')
    return acc, 0, 0, f1


def to_tensor(X):
    if isinstance(X, torch.Tensor):
        return X
    elif sp.issparse(X):
        X = X.toarray()
    elif isinstance(X, np.matrix):
        X = np.array(X)
    return torch.tensor(X, dtype=torch.float32)


def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    return nn.Parameter(torch.rand(shape) * 2 * bound - bound, requires_grad=True)


def get_Laplacian(A):
    device = A.device
    L = A + torch.eye(A.shape[0], device=device)
    D = L.sum(dim=1)
    sqrt_D = D.pow(-0.5)
    return sqrt_D.unsqueeze(1) * L * sqrt_D.unsqueeze(0)


def construct_knn_graph(features, k, device):
    """基于节点嵌入构建对称的 KNN 图（无向）。"""
    print(f"\nConstructing new graph using KNN (k={k})...")
    features_np = features.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features_np)
    adj_matrix = nbrs.kneighbors_graph(features_np).toarray()
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)  # 无向
    return to_tensor(adj_matrix).to(device)


# ----------------- 主模型 -----------------

class C_MSGFC(nn.Module):
    def __init__(self, X, A, labels, alpha, layers=None, acts=None, max_epoch=200, max_iter=50,
                 learning_rate=1e-3, coeff_reg=1e-4,
                 gat_heads=8,
                 beta_center=0.1, beta_membership=0.1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.X = to_tensor(X).to(self.device)
        self.A = to_tensor(A).to(self.device)
        self.labels = to_tensor(labels).to(self.device) if labels is not None else None
        self.n_clusters = int(self.labels.unique().shape[0]) if labels is not None else 0
        self.alpha = alpha
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg
        self.gat_heads = gat_heads
        self.beta_center = beta_center
        self.beta_membership = beta_membership

        self.input_dim = self.X.shape[1]
        self.layers = layers if layers else [256, 128]
        self.acts = acts if acts else [nn.LeakyReLU(0.2, inplace=True)] * (len(self.layers) - 1) + [None]
        self.data_size = self.X.shape[0]

        # Stage 间对齐的参考 (来自 Stage 1 的 Z1)
        self.U_prev = None  # N x C
        self.V_prev = None  # C x d

        self._build_up()
        self.to(self.device)

        self.U = None
        self.V = None
        self.embedding = None

    # ----- 网络搭建 -----
    def _build_up(self):
        # GAE: 类 GCN 的线性传播权重
        self.gae_weights = nn.ParameterList([
            get_weight_initial([self.input_dim, self.layers[0]])
        ] + [
            get_weight_initial([self.layers[i], self.layers[i + 1]])
            for i in range(len(self.layers) - 1)
        ])

        # GAT 分支
        self.gat_conv = GATConv(self.input_dim, self.layers[-1], heads=self.gat_heads, concat=True, dropout=0.6)

        # 维度对齐：将 GAE 与 GAT 的输出都映射到相同维度 d (= layers[-1])
        d = self.layers[-1]
        self.proj_gae = nn.Linear(d, d)
        self.proj_gat = nn.Linear(d * self.gat_heads, d)

        # 显式门控：输入为 [z_gae_proj || z_gat_proj]，输出为逐维 gate ∈ [0,1]，大小为 d
        self.W_g = nn.Linear(2 * d, d)

    # ----- 显式门控融合 -----
    def gated_fusion(self, z_gae, z_gat):
        # 维度对齐
        z_gae_p = self.proj_gae(z_gae)
        z_gat_p = self.proj_gat(z_gat)
        # gate 计算
        z_concat = torch.cat([z_gae_p, z_gat_p], dim=1)
        gate = torch.sigmoid(self.W_g(z_concat))
        # 逐维加权融合
        z_fused = gate * z_gae_p + (1.0 - gate) * z_gat_p
        return z_fused

    # ----- 前向 -----
    def forward(self, Laplacian):
        # GAE 分支
        h = self.X
        for i, w in enumerate(self.gae_weights):
            h = Laplacian @ (h @ w)
            if self.acts[i]:
                h = self.acts[i](h)
        emb_gae = h  # [N, d]

        # GAT 分支
        edge_index, _ = dense_to_sparse(self.A)
        emb_gat = self.gat_conv(self.X, edge_index)  # [N, d * heads]

        # 门控融合 + 归一化
        emb_fused = self.gated_fusion(emb_gae, emb_gat)
        emb_fused = emb_fused / emb_fused.norm(dim=1, keepdim=True).clamp(min=1e-7)
        self.embedding = emb_fused

        # 重构邻接（内积）
        return self.embedding @ self.embedding.T

    # ----- FCM 相关 -----
    def initialize_fcm(self):
        Z_np = self.embedding.detach().cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=0).fit(Z_np)
        initial_centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=self.device)
        self.V = nn.Parameter(initial_centers)
        self.U = self.update_U(self.embedding, self.V, tau=1.0)

    def update_U(self, Z, V, m=2.0, tau=1.0):
        Z_expand = Z.unsqueeze(1)               # [N, 1, d]
        V_expand = V.unsqueeze(0)               # [1, C, d]
        dist = ((Z_expand - V_expand) ** 2).sum(dim=2) + 1e-8  # [N, C]
        dist = dist / tau
        power = 1.0 / (m - 1)
        temp = (dist.unsqueeze(2) / dist.unsqueeze(1)) ** power  # [N, C, C]
        denom = temp.sum(dim=2)                                  # [N, C]
        U = 1.0 / denom
        U = torch.clamp(U, min=1e-5, max=1.0)
        return U / U.sum(dim=1, keepdim=True)

    def update_V(self, Z, U, m=2.0):
        U_m = U ** m
        num = U_m.T @ Z
        denom = U_m.sum(dim=0).unsqueeze(1) + 1e-8
        return num / denom

    # ----- 正则项 -----
    def build_loss_reg(self):
        loss_reg = sum(w.pow(2).sum() for w in self.gae_weights)
        loss_reg += sum(p.pow(2).sum() for p in self.gat_conv.parameters())
        loss_reg += sum(p.pow(2).sum() for p in self.proj_gae.parameters())
        loss_reg += sum(p.pow(2).sum() for p in self.proj_gat.parameters())
        loss_reg += sum(p.pow(2).sum() for p in self.W_g.parameters())
        return loss_reg

    # ----- 损失函数 -----
    def build_loss(self, recons_A, epoch=0):
        epsilon = 1e-8
        recons_A_no_diag = recons_A - torch.diag(torch.diag(recons_A))
        pos_weight = (self.data_size ** 2 - self.A.sum()) / (self.A.sum() + epsilon)

        # BCE 重构
        loss_1 = pos_weight * self.A.mul(torch.log(torch.clamp(recons_A_no_diag, min=epsilon))) + \
                 (1 - self.A).mul(torch.log(torch.clamp(1 - recons_A_no_diag, min=epsilon)))
        loss_1 = -loss_1.sum() / (self.data_size ** 2)

        # Trace FCM 聚类项
        Z = self.embedding
        m = 2.0
        U_m = self.U ** m
        D_cluster = torch.diag(U_m.sum(dim=1))
        A_cluster = torch.diag(U_m.sum(dim=0))
        term1 = torch.trace(Z.T @ D_cluster @ Z)
        term2 = 2 * torch.trace(Z.T @ U_m @ self.V)
        term3 = torch.trace(self.V.T @ A_cluster @ self.V)
        loss_2 = (term1 - term2 + term3) / self.data_size

        # 谱正则
        d = self.A.sum(1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(d + epsilon))
        L_sym = torch.eye(self.data_size, device=self.device) - D_inv_sqrt @ self.A @ D_inv_sqrt
        loss_spectral = torch.trace(Z.T @ L_sym @ Z) / self.data_size

        # 权重衰减
        loss_reg = self.coeff_reg * self.build_loss_reg()

        # 动态 alpha
        alpha_dynamic = self.alpha * min(1.0, (epoch + 1) / 100)

        total_loss = loss_1 + alpha_dynamic * loss_2 + 0.005 * loss_spectral + loss_reg

        # 阶段一致性损失：若存在上一阶段参考(U_prev, V_prev)，则加入
        if (self.U_prev is not None) and (self.V_prev is not None) and (self.U is not None) and (self.V is not None):
            cons = self.consistency_loss(self.V_prev, self.U_prev, self.V, self.U,
                                         beta_center=self.beta_center,
                                         beta_membership=self.beta_membership)
            total_loss = total_loss + cons
        return total_loss

    def build_pretrain_loss(self, recons_A):
        epsilon = 1e-8
        recons_A_no_diag = recons_A - torch.diag(torch.diag(recons_A))
        pos_weight = (self.data_size ** 2 - self.A.sum()) / (self.A.sum() + epsilon)
        loss = pos_weight * self.A.mul(torch.log(torch.clamp(recons_A_no_diag, min=epsilon))) + \
               (1 - self.A).mul(torch.log(torch.clamp(1 - recons_A_no_diag, min=epsilon)))
        loss = -loss.sum() / (self.data_size ** 2)
        loss += self.coeff_reg * self.build_loss_reg()
        return loss

    # ----- 一致性损失 -----
    @staticmethod
    def consistency_loss(V_prev, U_prev, V_curr, U_curr, beta_center=0.1, beta_membership=0.1):
        loss_center = F.mse_loss(V_curr, V_prev)
        loss_membership = F.mse_loss(U_curr, U_prev)
        return beta_center * loss_center + beta_membership * loss_membership

    # ----- Stage 1：仅重构，得到 Z1，并初始化上一阶段的(U_prev, V_prev) -----
    def train_stage1(self, epochs, learning_rate=None, init_fcm_with_kmeans=True):
        print(f'====== STAGE 1: Learning Initial Representation (Z₁) ======')
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate or self.learning_rate)
        Laplacian = get_Laplacian(self.A)
        for i in range(epochs):
            optimizer.zero_grad()
            recons_A = self(Laplacian)
            loss = self.build_pretrain_loss(recons_A)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Stage 1 - Epoch {i + 1}/{epochs}, Reconstruction Loss: {loss.item():.4f}')
        print('Stage 1 training finished.')

        # 基于 Z1 初始化上一阶段的 U_prev, V_prev（用于 Stage 2 一致性约束）
        with torch.no_grad():
            Z1 = self.embedding.detach()
            # KMeans 产生初始中心，再用一次 U/V 更新得到更稳的参考
            if init_fcm_with_kmeans:
                Z_np = Z1.cpu().numpy()
                km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=0).fit(Z_np)
                V_prev = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=self.device)
            else:
                # 随机选择中心（不推荐）
                idx = torch.randperm(Z1.size(0))[:self.n_clusters]
                V_prev = Z1[idx]
            U_prev = self.update_U(Z1, V_prev, tau=1.0)
            # 固定为 buffer（非参数，不参与优化）
            self.U_prev = U_prev.detach()
            self.V_prev = V_prev.detach()

        return self.embedding.detach()

    # ----- Stage 2：联合优化 + 一致性约束 -----
    def run(self, return_final_metrics=False, log_all_epochs=False,
            warmup_epoch=20,
            initial_tau=2.0, min_tau=0.3, decay_rate=0.95,
            learning_rate=None, momentum=0.9,
            best_model_path='best_model_stage2.pth'):
        print(f'\n====== STAGE 2: Joint Training with Refined Graph ======')

        history = []
        Laplacian = get_Laplacian(self.A)  # 注意：这里的 self.A 应当是精炼后的图 A₁
        lr = learning_rate or self.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[warmup_epoch], gamma=0.1)

        best_acc, best_nmi, best_ari, best_f1 = 0.0, 0.0, 0.0, 0.0
        best_epoch = 0
        patience = 20
        patience_counter = 0

        for epoch in range(self.max_epoch):
            tau = max(min_tau, initial_tau * (decay_rate ** (epoch / 2)))
            optimizer.zero_grad()
            recons_A = self(Laplacian)

            if epoch < warmup_epoch:
                # Warmup 仅监督重构，避免聚类项过早牵引
                loss = self.build_pretrain_loss(recons_A)
            else:
                if epoch == warmup_epoch:
                    print("Warmup finished. Initializing FCM clusters...")
                    self.initialize_fcm()

                # FCM 交替更新（带动量的中心）
                new_U = self.update_U(self.embedding, self.V, tau=tau)
                new_V = self.update_V(self.embedding, new_U)
                with torch.no_grad():
                    self.V.data = momentum * self.V.detach() + (1 - momentum) * new_V
                self.U = new_U

                loss = self.build_loss(recons_A, epoch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch >= warmup_epoch:
                acc, nmi, ari, f1 = self.evaluate_clustering()
                if log_all_epochs:
                    history.append({'epoch': epoch + 1, 'loss': loss.item(), 'acc': acc, 'nmi': nmi, 'ari': ari, 'f1': f1})
                print(f"Epoch {epoch + 1}/{self.max_epoch} summary: loss={loss.item():.4f}, "
                      f"ACC={acc * 100:.2f}, NMI={nmi * 100:.2f}, ARI={ari * 100:.2f}, F1={f1 * 100:.2f}")

                if acc > best_acc:
                    best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1
                    best_epoch = epoch + 1
                    patience_counter = 0
                    print(
                        f"🎉 New best ACC: {best_acc * 100:.2f} at epoch {best_epoch}. Saving model to '{best_model_path}'...")
                    torch.save(self.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"⏳ Early stopping at epoch {epoch + 1}. Best ACC was {best_acc * 100:.2f} at epoch {best_epoch}.")
                    break
            else:
                print(f"Warmup Epoch {epoch + 1}/{warmup_epoch} summary: loss={loss.item():.4f}")

            scheduler.step()

        print("\n" + "=" * 50)
        print(f"Training finished. Loading best model from '{best_model_path}'...")
        if os.path.exists(best_model_path):
            self.load_state_dict(torch.load(best_model_path))
        else:
            print(f"⚠️ Warning: Best model file '{best_model_path}' not found.")

        print(f"\n🏆 Final evaluation with best model (from epoch {best_epoch}):")
        print(f"   ACC: {best_acc * 100:.2f}")
        print(f"   NMI: {best_nmi * 100:.2f}")
        print(f"   ARI: {best_ari * 100:.2f}")
        print(f"   F1:  {best_f1 * 100:.2f}")
        print("=" * 50 + "\n")

        if return_final_metrics:
            return best_acc, best_nmi, best_ari, best_f1
        return None

    # ----- 评估 -----
    def evaluate_clustering(self):
        if self.U is None:
            return 0.0, 0.0, 0.0, 0.0
        pred_labels_raw = self.U.argmax(dim=1).cpu().numpy()
        true_labels = self.labels.cpu().numpy().astype(np.int64)
        return cal_clustering_metric(true_labels, pred_labels_raw)

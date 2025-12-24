"""
# visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import numpy as np
import os # 导入 os 模块

def visualize_embeddings(embedding, labels_true, labels_pred, title_prefix,
                         save_path=None, umap_n_neighbors=15, umap_min_dist=0.1):

    print(f"\n[Visualizer] --- Executing visualize_embeddings for: '{title_prefix}' ---")

    try:
        # --- UMAP 降维 ---
        print("[Visualizer] Starting UMAP dimensionality reduction with n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}...")
        reducer = UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embedding)
        print("[Visualizer] UMAP reduction complete.")

        # --- 绘图 ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'UMAP Visualization: {title_prefix}', fontsize=16)
        num_classes_true = len(np.unique(labels_true))
        palette_true = sns.color_palette("viridis", num_classes_true)

        # 1. 根据真实标签着色
        sns.scatterplot(
            x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_true.astype(int),
            palette=palette_true, legend='full', ax=axes[0], s=50, alpha=0.8
        )
        axes[0].set_title('Colored by Ground Truth Labels')
        axes[0].set_xlabel('UMAP Component 1')
        axes[0].set_ylabel('UMAP Component 2')
        axes[0].legend(title='True Class', bbox_to_anchor=(1.05, 1), loc=2)

        # 2. 根据预测标签着色
        num_classes_pred = len(np.unique(labels_pred))
        palette_pred = sns.color_palette("flare", num_classes_pred)
        sns.scatterplot(
            x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_pred.astype(int),
            palette=palette_pred, legend='full', ax=axes[1], s=50, alpha=0.8
        )
        axes[1].set_title('Colored by Predicted Cluster Labels')
        axes[1].set_xlabel('UMAP Component 1')
        axes[1].set_ylabel('UMAP Component 2')
        axes[1].legend(title='Predicted Cluster', bbox_to_anchor=(1.05, 1), loc=2)

        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        print("[Visualizer] Plot created successfully.")

        # --- 保存文件 ---
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                print(f"[Visualizer] Save directory '{save_dir}' does not exist. Creating it...")
                os.makedirs(save_dir)

            print(f"[Visualizer] Attempting to save plot to: {save_path}")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"✅ [Visualizer] Plot successfully saved to '{save_path}'")
        else:
            print("[Visualizer] No save_path provided. Showing plot directly.")
            plt.show()

    except Exception as e:
        print(f"❌ [Visualizer] An error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()
# visualize.py

# ... (保留你已有的 visualize_embeddings 函数) ...

def visualize_embeddings_3d(embedding, labels, title, save_path=None):

    print(f"\n[Visualizer 3D] --- Executing 3D visualization for: '{title}' ---")
    try:
        # --- UMAP 降维到 3D ---
        print("[Visualizer 3D] Starting UMAP reduction to 3 dimensions...")
        reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_3d = reducer.fit_transform(embedding)
        print("[Visualizer 3D] UMAP 3D reduction complete.")

        # --- 3D 绘图 ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d') # 创建一个3D坐标轴

        # 使用一个颜色盘
        num_classes = len(np.unique(labels))
        palette = sns.color_palette("viridis", num_classes)
        colors = [palette[i] for i in labels.astype(int)]

        ax.scatter(
            embedding_3d[:, 0], # X 坐标
            embedding_3d[:, 1], # Y 坐标
            embedding_3d[:, 2], # Z 坐标
            c=colors,           # 颜色
            s=20                # 点的大小
        )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')

        if save_path:
            print(f"[Visualizer 3D] Attempting to save 3D plot to: {save_path}")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"✅ [Visualizer 3D] Plot successfully saved to '{save_path}'")
        else:
            print("[Visualizer 3D] No save_path provided. Showing plot directly.")
            plt.show()

    except Exception as e:
        print(f"❌ [Visualizer 3D] An error occurred during 3D visualization: {e}")
        import traceback
        traceback.print_exc()
"""
# visualize.py
"""
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import TSNE  # <-- 新增导入
import numpy as np
import os


# --- 你现有的 UMAP 2D 可视化函数 (无需改动) ---
def visualize_embeddings(embedding, labels_true, labels_pred, title_prefix,
                         save_path=None, umap_n_neighbors=8, umap_min_dist=0.0001):
 
    print(f"\n[Visualizer UMAP-2D] --- Executing for: '{title_prefix}' ---")
    try:
        reducer = UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embedding)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'UMAP Visualization: {title_prefix}', fontsize=16)
        num_classes_true = len(np.unique(labels_true))
        palette_true = sns.color_palette("viridis", num_classes_true)
        sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_true.astype(int), palette=palette_true,
                        legend='full', ax=axes[0], s=50, alpha=0.8)
        axes[0].set_title('Colored by Ground Truth Labels')
        axes[0].set_xlabel('UMAP Component 1')
        axes[0].set_ylabel('UMAP Component 2')
        axes[0].legend(title='True Class', bbox_to_anchor=(1.05, 1), loc=2)
        num_classes_pred = len(np.unique(labels_pred))
        palette_pred = sns.color_palette("flare", num_classes_pred)
        sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_pred.astype(int), palette=palette_pred,
                        legend='full', ax=axes[1], s=50, alpha=0.8)
        axes[1].set_title('Colored by Predicted Cluster Labels')
        axes[1].set_xlabel('UMAP Component 1')
        axes[1].set_ylabel('UMAP Component 2')
        axes[1].legend(title='Predicted Cluster', bbox_to_anchor=(1.05, 1), loc=2)
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            print(f"[Visualizer UMAP-2D] Attempting to save plot to: {save_path}")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"✅ [Visualizer UMAP-2D] Plot successfully saved.")
        else:
            plt.show()
    except Exception as e:
        print(f"❌ [Visualizer UMAP-2D] An error occurred: {e}")


# --- 你现有的 UMAP 3D 可视化函数 (无需改动) ---
def visualize_embeddings_3d(embedding, labels, title, save_path=None):
    
    print(f"\n[Visualizer UMAP-3D] --- Executing for: '{title}' ---")
    try:
        reducer = UMAP(n_components=3, n_neighbors=5, min_dist=0.001, random_state=42)
        embedding_3d = reducer.fit_transform(embedding)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        num_classes = len(np.unique(labels))
        palette = sns.color_palette("viridis", num_classes)
        colors = [palette[i] for i in labels.astype(int)]
        ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], c=colors, s=20)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        if save_path:
            print(f"[Visualizer UMAP-3D] Attempting to save 3D plot to: {save_path}")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"✅ [Visualizer UMAP-3D] Plot successfully saved.")
        else:
            plt.show()
    except Exception as e:
        print(f"❌ [Visualizer UMAP-3D] An error occurred: {e}")


# --- 新增: t-SNE 可视化函数 ---
def visualize_embeddings_tsne(embedding, labels_true, labels_pred, title_prefix,
                              save_path=None, perplexity=50):
    
    print(f"\n[Visualizer t-SNE] --- Executing for: '{title_prefix}' ---")
    try:
        # --- t-SNE 降维 ---
        print(f"[Visualizer t-SNE] Starting t-SNE reduction with perplexity={perplexity}...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42, init='pca',
                    learning_rate='auto')
        embedding_2d = tsne.fit_transform(embedding)
        print("[Visualizer t-SNE] t-SNE reduction complete.")

        # --- 绘图 (逻辑与UMAP可视化几乎一样) ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f't-SNE Visualization: {title_prefix}', fontsize=16)

        # 1. 根据真实标签着色
        num_classes_true = len(np.unique(labels_true))
        palette_true = sns.color_palette("viridis", num_classes_true)
        sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_true.astype(int), palette=palette_true,
                        legend='full', ax=axes[0], s=50, alpha=0.8)
        axes[0].set_title('Colored by Ground Truth Labels')
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        axes[0].legend(title='True Class', bbox_to_anchor=(1.05, 1), loc=2)

        # 2. 根据预测标签着色
        num_classes_pred = len(np.unique(labels_pred))
        palette_pred = sns.color_palette("flare", num_classes_pred)
        sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_pred.astype(int), palette=palette_pred,
                        legend='full', ax=axes[1], s=50, alpha=0.8)
        axes[1].set_title('Colored by Predicted Cluster Labels')
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        axes[1].legend(title='Predicted Cluster', bbox_to_anchor=(1.05, 1), loc=2)

        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            print(f"[Visualizer t-SNE] Attempting to save plot to: {save_path}")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"✅ [Visualizer t-SNE] Plot successfully saved.")
        else:
            plt.show()
    except Exception as e:
        print(f"❌ [Visualizer t-SNE] An error occurred: {e}")"""
# visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def get_palette(n_classes):
    """统一调色板"""
    if n_classes <= 10:
        return sns.color_palette("tab10", n_classes)
    elif n_classes <= 20:
        return sns.color_palette("tab20", n_classes)
    else:
        return sns.color_palette("hls", n_classes)


def visualize_embeddings(embedding, labels_true, labels_pred, title_prefix, save_path=None):
    """
    2D UMAP 可视化（对比真实标签 vs 预测标签）
    """
    reducer = UMAP(
        n_neighbors=15,  # 更大，保留全局结构
        min_dist=0.05,     # 更大，簇分离更清晰
        n_components=2,
        random_state=42,
    )
    emb_2d = reducer.fit_transform(embedding)

    n_classes = len(np.unique(labels_true))
    palette = get_palette(n_classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Ground Truth ---
    sns.scatterplot(
        x=emb_2d[:, 0], y=emb_2d[:, 1],
        hue=labels_true, palette=palette,
        legend="full", s=20, ax=axes[0], linewidth=0
    )
    axes[0].set_title(f"{title_prefix} - Ground Truth")

    # --- Predicted ---
    sns.scatterplot(
        x=emb_2d[:, 0], y=emb_2d[:, 1],
        hue=labels_pred, palette=palette,
        legend="full", s=20, ax=axes[1], linewidth=0
    )
    axes[1].set_title(f"{title_prefix} - Predicted")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def visualize_embeddings_3d(embedding, labels, title, save_path=None):
    """
    3D UMAP 可视化（真实标签）
    """
    reducer = UMAP(
        n_neighbors=100,
        min_dist=0.2,
        n_components=3,
        random_state=42,
    )
    emb_3d = reducer.fit_transform(embedding)

    n_classes = len(np.unique(labels))
    palette = get_palette(n_classes)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2],
        c=[palette[l] for l in labels],
        s=15, alpha=0.8
    )
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def visualize_embeddings_tsne(embedding, labels_true, labels_pred, title_prefix, save_path=None, perplexity=100):
    """
    2D t-SNE 可视化（对比真实标签 vs 预测标签）
    """
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=2000,
        random_state=42,
        init="pca"
    )
    emb_2d = reducer.fit_transform(embedding)

    n_classes = len(np.unique(labels_true))
    palette = get_palette(n_classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Ground Truth ---
    sns.scatterplot(
        x=emb_2d[:, 0], y=emb_2d[:, 1],
        hue=labels_true, palette=palette,
        legend="full", s=20, ax=axes[0], linewidth=0
    )
    axes[0].set_title(f"{title_prefix} - Ground Truth")

    # --- Predicted ---
    sns.scatterplot(
        x=emb_2d[:, 0], y=emb_2d[:, 1],
        hue=labels_pred, palette=palette,
        legend="full", s=20, ax=axes[1], linewidth=0
    )
    axes[1].set_title(f"{title_prefix} - Predicted")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# File: run.py
# Description: Hyperparameter search script for the TWO-STAGE EGAE-GAT model.

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools
import os
import time
import seaborn as sns
import traceback

# 关键：从您的模型文件中导入两个阶段模型和KNN构图函数
# 请确保您的模型文件名与此处匹配（例如，egae_gat_two_stage.py）
from model import C_MSGFC, construct_knn_graph
from utils import load_data

warnings.filterwarnings('ignore')

# 全局列表，用于存储所有实验的结果
results = []


def run_single_experiment(
        # 新增：第一阶段和图精炼的参数
        stage1_epochs,
        knn_k,
        # 第二阶段的参数 (原始参数)
        alpha,
        coeff_reg,
        lr,
        layers,
        activation_func,
        max_epoch,
        # 通用参数
        dataset_name,
        save_dir):
    """
    运行一个完整的、遵循双阶段架构的训练实验。
    """
    # 创建一个对文件名安全配置字符串
    config_str_safe = (f"s1ep{stage1_epochs}_knn{knn_k}_alpha{alpha}_reg{coeff_reg}_lr{lr}_"
                       f"layers{'-'.join(map(str, layers))}_act{activation_func.__name__}")

    print(f"\n{'=' * 80}\n🚀 Running experiment: {config_str_safe} on '{dataset_name}'\n{'=' * 80}")

    try:
        model_save_dir = os.path.join(save_dir, 'saved_models')
        os.makedirs(model_save_dir, exist_ok=True)
        # 第二阶段模型的保存路径
        best_model_path = os.path.join(model_save_dir, f'best_model_{config_str_safe}.pth')

        features, adjacency, labels = load_data(dataset_name)
        acts = [activation_func] * (len(layers) - 1) + [None]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        start_time = time.time()

        # --- 步骤 1: 训练第一阶段模型以学习初始表示 Z₁ ---
        print("--- Step 1: Training Stage 1 Model to get Z₁ ---")
        model_stage1 = C_MSGFC(
            X=features, A=adjacency, labels=labels, alpha=0,  # Alpha为0，不进行聚类
            layers=layers, acts=acts, learning_rate=lr, coeff_reg=coeff_reg, device=device
        )
        # 使用第一阶段专属的训练函数
        z1_embedding = model_stage1.train_stage1(epochs=stage1_epochs, learning_rate=lr)

        # --- 步骤 2: 使用 Z₁ 进行图结构精炼 (KNN) ---
        print(f"--- Step 2: Refining graph structure with KNN (k={knn_k}) ---")
        refined_adjacency = construct_knn_graph(features=z1_embedding, k=knn_k, device=device)

        # --- 步骤 3: 训练第二阶段模型，使用精炼后的图 ---
        print("--- Step 3: Training Stage 2 Model with the refined graph ---")
        model_stage2 = C_MSGFC(
            X=features, A=refined_adjacency, labels=labels, alpha=alpha,  # 使用精炼图和指定的alpha
            layers=layers, acts=acts, max_epoch=max_epoch,
            learning_rate=lr, coeff_reg=coeff_reg, device=device
        )

        # 调用完整的联合优化训练流程 `run`
        # 修正：run方法现在正确返回4个值
        acc, nmi, ari, f1 = model_stage2.run(
            return_final_metrics=True,
            best_model_path=best_model_path,
            warmup_epoch=10  # 可以设为一个超参数
        )
        training_time = time.time() - start_time

        results.append({
            'dataset': dataset_name, 'stage1_epochs': stage1_epochs, 'knn_k': knn_k,
            'alpha': alpha, 'coeff_reg': coeff_reg, 'lr': lr,
            'layers': str(layers), 'activation': activation_func.__name__,
            'ACC': acc, 'NMI': nmi, 'ARI': ari, 'F1': f1,
            'training_time': training_time, 'status': 'Success'
        })

    except Exception as e:
        print(f"⚠️ An error occurred during experiment: {config_str_safe}")
        traceback.print_exc()
        results.append({
            'dataset': dataset_name, 'stage1_epochs': stage1_epochs, 'knn_k': knn_k,
            'alpha': alpha, 'coeff_reg': coeff_reg, 'lr': lr,
            'layers': str(layers), 'activation': activation_func.__name__,
            'ACC': np.nan, 'NMI': np.nan, 'ARI': np.nan, 'F1': np.nan,
            'training_time': np.nan, 'status': f'Error: {e}'
        })


def analyze_and_save_summary(df, save_dir, dataset_name):
    """
    分析实验结果，打印总结并保存到文件。
    （此函数已更新以显示新的超参数）
    """
    summary_path = os.path.join(save_dir, 'best_configs_summary.txt')

    df_success = df[df['status'] == 'Success'].copy()
    if df_success.empty:
        message = "No successful runs were completed to analyze."
        print(message)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(message)
        return

    summary_lines = []
    header = "=" * 80
    title = f"🏆 Best Configurations Found for '{dataset_name}' (Two-Stage Model)"
    summary_lines.append(header)
    summary_lines.append(title)

    metrics_to_analyze = ['ACC', 'NMI', 'ARI', 'F1']
    for metric in metrics_to_analyze:
        df_success[metric] = pd.to_numeric(df_success[metric], errors='coerce')
        if not df_success[metric].dropna().empty:
            best_config = df_success.loc[df_success[metric].idxmax()]
            summary_lines.append(f"\n⭐ Best by {metric}: {best_config[metric]:.4f}")
            # 从要打印的系列中删除不必要的信息
            cols_to_drop = ['status', 'training_time', 'dataset']
            summary_lines.append(best_config.drop(cols_to_drop).to_string())

    summary_lines.append("\n" + header)
    final_summary = "\n".join(summary_lines)

    print("\n" + final_summary)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(final_summary)
    print(f"\n📄 Summary saved to '{summary_path}'")


def plot_search_results(df, save_dir):
    """
    生成并保存可视化超参数影响的图表。
    （此函数已更新以绘制新的超参数）
    """
    print("\n📊 Plotting search results...")
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    df_success = df[df['status'] == 'Success'].copy()
    if df_success.empty:
        print("No successful experiments to plot.")
        return

    # 确保列是正确的数值类型
    for col in ['alpha', 'lr', 'knn_k', 'stage1_epochs', 'ACC', 'NMI', 'ARI', 'F1']:
        df_success[col] = pd.to_numeric(df_success[col], errors='coerce')

    metrics_to_plot = ['ACC', 'NMI', 'ARI', 'F1']
    # 将新参数加入分析列表
    params_to_analyze = ['alpha', 'lr', 'layers', 'activation', 'knn_k', 'stage1_epochs']

    for metric in metrics_to_plot:
        for param in params_to_analyze:
            plt.figure(figsize=(12, 7))
            # 使用箱线图或点图更适合分类和整数参数
            if param in ['layers', 'activation']:
                sns.boxplot(data=df_success, x=param, y=metric)
            else:
                sns.lineplot(data=df_success, x=param, y=metric, marker='o', errorbar='sd')

            plt.title(f'{metric} vs {param.capitalize()} (Two-Stage Model)')
            plt.ylabel(metric)
            plt.xlabel(param.capitalize())
            plt.grid(True, which='both', linestyle='--')
            if param in ['layers', 'activation']:
                plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{metric}_vs_{param}.png'))
            plt.close()
    print(f"Plots saved to '{plot_dir}'")


def main():
    """
    主函数，定义超参数空间并运行搜索。
    """
    # --- 核心配置 ---
    dataset_name = 'hhar'  # 在此切换数据集: 'cora', 'citeseer', 'acm', 'dblp'
    save_dir = f'hyperparam_search_2stage_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(save_dir, exist_ok=True)

    # --- 定义扩展后的超参数搜索空间 ---
    # 阶段一和图精炼的参数
    stage1_epochs_list = [100,150]
    knn_k_list = [10, 15, 20]

    # 阶段二的参数
    alpha_list = [5, 10, 20]
    coeff_reg_list = [1e-5, 1e-4, 1e-3]
    lr_list = [0.0005, 0.001, 0.005]
    max_epoch_list = [200]  # 第二阶段的总轮数
    layer_options = [[256, 128], [256, 128, 64]]
    activation_options = [F.relu, torch.tanh]

    # 使用 itertools.product 创建所有可能的组合
    search_space = list(itertools.product(
        stage1_epochs_list,
        knn_k_list,
        alpha_list,
        coeff_reg_list,
        lr_list,
        layer_options,
        activation_options,
        max_epoch_list
    ))

    total_experiments = len(search_space)
    print(f"🔬 Starting two-stage hyperparameter search for '{dataset_name}' with {total_experiments} combinations...")

    total_start_time = time.time()

    for i, params in enumerate(search_space):
        print(f"\n--- Running Combination {i + 1}/{total_experiments} ---")
        run_single_experiment(
            stage1_epochs=params[0],
            knn_k=params[1],
            alpha=params[2],
            coeff_reg=params[3],
            lr=params[4],
            layers=params[5],
            activation_func=params[6],
            max_epoch=params[7],
            dataset_name=dataset_name,
            save_dir=save_dir
        )

    total_time_taken = time.time() - total_start_time
    print(f"\n✅ Hyperparameter search completed in {total_time_taken / 3600:.2f} hours.")

    # --- 保存、分析和绘制结果 ---
    df_results = pd.DataFrame(results)
    df_results.to_excel(os.path.join(save_dir, 'full_search_results_2stage.xlsx'), index=False)

    analyze_and_save_summary(df_results, save_dir, dataset_name)
    plot_search_results(df_results, save_dir)


if __name__ == '__main__':
    main()
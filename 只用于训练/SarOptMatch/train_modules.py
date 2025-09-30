import os
import glob
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import SarOptMatch
import SarOptMatch.evaluation
import SarOptMatch.visualization


# === Top-5 Precision/Recall Metric ===
def top5_precision_recall(y_true, y_pred_logits, top_k=5):
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred_logits, [tf.shape(y_pred_logits)[0], -1])
    ground_truth_indices = tf.argmax(y_true_flat, axis=-1, output_type=tf.int32)[:, None]
    top_k_values, top_k_indices = tf.nn.top_k(y_pred_flat, k=top_k)
    correct_in_topk = tf.reduce_any(tf.equal(top_k_indices, ground_truth_indices), axis=-1)
    precision_at_topk = tf.reduce_mean(tf.cast(correct_in_topk, tf.float32))
    recall_at_topk = precision_at_topk  # Since ground truth is single-class
    return precision_at_topk, recall_at_topk


# === Best Model Saver Callback ===
class BestModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, matcher, dataset_name):
        super().__init__()
        self.validation_data = validation_data
        self.matcher = matcher
        self.dataset_name = dataset_name
        self.best_score = -np.inf
        self.score_history = []  # ✅ 新增：用于存储每轮分数

    def on_epoch_end(self, epoch, logs=None):
        heatmaps = self.matcher.predict_heatmap(self.validation_data)
        euc_dists = SarOptMatch.evaluation.print_results(self.validation_data, heatmaps)
        acc = {
            k: round(np.mean(euc_dists <= int(k[0])), 4)
            for k in ["1px", "2px", "3px", "5px"]
        }
        avg_euc_dist = round(np.mean(euc_dists), 4)

        score = round(acc["1px"] + acc["2px"] + acc["3px"] + acc["5px"] - avg_euc_dist / 50, 4)
        self.score_history.append(score)  # ✅ 仅记录

        if score > self.best_score:
            self.best_score = score
            os.makedirs("lambda_models", exist_ok=True)
            save_path = f"lambda_models/{self.dataset_name}_best_model.h5"
            self.matcher.model.save_weights(save_path)
            print(f"\n📏 New best model saved | score = {score:.4f}")
        else:
            print(f"-- No improvement | score = {score:.4f} | best = {self.best_score:.4f}")

# === Final Evaluation: Load Best Model, Predict, Visualize ===
# def evaluate_best_model(matcher, validation_data, validation_dataRGB, dataset_name):
#     print("\n-- Loading best model for final evaluation --")
#     model_path = f"lambda_models/{dataset_name}_best_model.h5"
#     if not os.path.exists(model_path):
#         print(f"⚠️ 未找到最佳模型: {model_path}，无法评估")
#         return
#
#     print(f"✅ 找到并加载最优模型: {model_path}")
#     matcher.model.load_weights(model_path)
#
#     heatmaps = matcher.predict_heatmap(validation_data)
#     SarOptMatch.evaluation.print_results(validation_data, heatmaps)
#
#     feature_maps = matcher.calculate_features(validation_data)
#     SarOptMatch.visualization.visualize_dataset_with_GUI(
#         validation_dataRGB, heatmaps, feature_maps, dataset_name
#     )


def evaluate_best_model(matcher, validation_data, validation_dataRGB, dataset_name):
    print("\n-- Loading best model for final evaluation --")
    model_path = f"lambda_models/{dataset_name}_best_model.h5"

    # === 增加判断：尝试加载最佳模型 ===
    if os.path.exists(model_path):
        print(f"✅ 找到并加载最优模型: {model_path}")
        matcher.model.load_weights(model_path)
    else:
        print(f"⚠️ 未找到最佳模型: {model_path}，将直接使用当前已加载的模型进行评估")

    try:
        heatmaps = matcher.predict_heatmap(validation_data)
        SarOptMatch.evaluation.print_results(validation_data, heatmaps)

        feature_maps = matcher.calculate_features(validation_data)
        SarOptMatch.visualization.visualize_dataset_with_GUI(
            validation_dataRGB, heatmaps, feature_maps, dataset_name
        )
    except Exception as e:
        print(f"⚠️ 评估过程出错: {str(e)}")


import re

def extract_alpha_threshold(model_name):
    match = re.search(r"_a([\d.]+)_t([\d.]+)", model_name)
    if match:
        alpha = match.group(1)
        threshold = match.group(2)
        return alpha, threshold
    return None, None


# === 绘制训练/验证损失曲线 ===
def plot_training_curve(model_name):
    history_path = f"weights/{model_name}_history"
    if not os.path.exists(history_path):
        print(f"⚠️ 未找到历史文件: {history_path}")
        return

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])

    if not loss or not val_loss:
        print("⚠️ 历史记录为空，无法绘图。")
        return

    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    score = history.get('score', [])

    if not loss or not val_loss:
        print("⚠️ 历史记录为空，无法绘图。")
        return

    epochs_range = range(1, len(loss) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ✅ 左轴：损失
    ax1.plot(epochs_range, loss, label='Train Loss', color='tab:blue', marker='o')
    ax1.plot(epochs_range, val_loss, label='Val Loss', color='tab:orange', marker='^')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # ✅ 右轴：得分
    if score:
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, score, label='Score', color='tab:green', marker='s')
        ax2.set_ylabel('Evaluation Score')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.legend(loc='upper right')


    # === 提取 alpha 和 threshold 参数 ===
    alpha, threshold = extract_alpha_threshold(model_name)
    param_text = f"(α={alpha}, τ={threshold})" if alpha and threshold else ""

    # === 添加标题与图注 ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.title(f'{model_name} | Loss & Score Curve {param_text} ({timestamp})')
    ax1.text(0.02, 0.95,
             f'alpha = {alpha}\nthreshold = {threshold}',
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    save_path = f"weights/{model_name}_curve_{timestamp}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 曲线图已保存到: {save_path}")




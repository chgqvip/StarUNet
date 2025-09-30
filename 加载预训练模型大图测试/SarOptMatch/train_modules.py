# train_modules.py

import os
import glob  # 如果需要通配符匹配文件
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




from . import evaluation as SarOptMatch_evaluation
from . import visualization as SarOptMatch_visualization


# === Top-5 Precision/Recall Metric ===
# (代码保持不变)
def top5_precision_recall(y_true, y_pred_logits, top_k=5):
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred_logits, [tf.shape(y_pred_logits)[0], -1])
    ground_truth_indices = tf.argmax(y_true_flat, axis=-1, output_type=tf.int32)[:, None]
    top_k_values, top_k_indices = tf.nn.top_k(y_pred_flat, k=top_k)
    correct_in_topk = tf.reduce_any(tf.equal(top_k_indices, ground_truth_indices), axis=-1)
    precision_at_topk = tf.reduce_mean(tf.cast(correct_in_topk, tf.float32))
    recall_at_topk = precision_at_topk
    return precision_at_topk, recall_at_topk


# === Best Model Saver Callback ===
class BestModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, matcher, dataset_name):
        super().__init__()
        self.validation_data = validation_data
        self.matcher = matcher  # 这是一个 SarOptMatch.architectures.SAR_opt_Matcher 实例
        self.dataset_name = dataset_name
        self.best_score = -np.inf
        self.score_history = []

    def on_epoch_end(self, epoch, logs=None):
        # 假设 matcher 有 predict_heatmap 方法
        # 并且 SarOptMatch.evaluation.print_results 返回 euc_dists
        # 注意：在回调中进行重量级评估（如 predict_heatmap）可能会显著减慢训练速度
        # 考虑是否可以从 logs 中获取验证指标，或者减少评估频率

        # 确保 self.matcher.model 存在
        if self.matcher.model is None:
            print("⚠️ [BestModelSaver] Matcher model is None, skipping score calculation.")
            if logs:  # 尝试从logs中获取一些信息
                logs['score'] = -np.inf  # 或者其他默认值
            return

        # 假设 self.matcher 有 predict_heatmap 方法
        # 这个方法需要能处理 self.validation_data
        # 并且返回热图
        # 注意: predict_heatmap 可能会消耗大量时间和资源
        print(f"\n[BestModelSaver] Epoch {epoch + 1}: Calculating score for model '{self.matcher.model_name}'...")
        try:
            # 确保 validation_data 是以合适的批次大小准备的
            # predict_heatmap 内部应该能高效处理
            heatmaps = self.matcher.predict_heatmap(self.validation_data)  # 假设此方法存在

            # 假设 SarOptMatch.evaluation.print_results 存在并返回欧氏距离
            # 它也需要 validation_data 来获取真实偏移

            euc_dists = SarOptMatch_evaluation.print_results(self.validation_data, heatmaps)
        except Exception as e:
            print(f"⚠️ [BestModelSaver] Error during heatmap prediction or result printing: {e}")
            import traceback
            traceback.print_exc()
            if logs:
                logs['score'] = -np.inf  # 记录一个无效分数
            self.score_history.append(-np.inf)  # 记录无效分数
            return

        acc = {
            k: round(np.mean(euc_dists <= int(k[0])), 4)
            for k in ["1px", "2px", "3px", "5px"]
        }
        avg_euc_dist = round(np.mean(euc_dists), 4)

        # 计算分数
        score = round(acc["1px"] + acc["2px"] + acc["3px"] + acc["5px"] - avg_euc_dist / 50, 4)
        self.score_history.append(score)
        print(f"[BestModelSaver] Epoch {epoch + 1}: Calculated score = {score:.4f}")

        if logs is not None:
            logs['score'] = score  # 将分数添加到logs中，这样Keras的History对象可能会记录它

        if score > self.best_score:
            self.best_score = score
            # 确保 lambda_models 目录存在
            os.makedirs("lambda_models", exist_ok=True)
            # 构建保存路径
            save_path = f"lambda_models/{self.dataset_name}_best_model.h5"
            try:
                self.matcher.model.save_weights(save_path)
                print(f"📏 [BestModelSaver] New best model saved to {save_path} | score = {score:.4f}")
            except Exception as e:
                print(f"⚠️ [BestModelSaver] Failed to save best model weights: {e}")
        else:
            print(f"-- [BestModelSaver] No improvement | score = {score:.4f} | best = {self.best_score:.4f}")

    # (可选) 如果想让 score_history 与 Keras History 一起保存
    # def on_train_end(self, logs=None):
    #     if hasattr(self.model, 'history') and self.model.history is not None:
    #         if 'score' not in self.model.history.history : # 检查是否已被logs['score']=score添加
    #              self.model.history.history['score_from_saver'] = self.score_history
    #         print("[BestModelSaver] Score history potentially added to model's history object.")


# === Final Evaluation: Load Best Model, Predict, Visualize ===
def evaluate_best_model(matcher, validation_data, validation_dataRGB, dataset_name,extract_features=False):
    # (代码基本不变，但需要确保 matcher.calculate_features 接受正确的参数)
    print("\n-- Loading best model for final evaluation --")
    model_path = f"lambda_models/{dataset_name}_best_model.h5"
    if not os.path.exists(model_path):
        print(f"⚠️ 未找到最佳模型: {model_path}，无法评估")
        return

    print(f"✅ 找到并加载最优模型: {model_path}")
    try:
        matcher.model.load_weights(model_path)
    except Exception as e:
        print(f"❌ 加载最佳模型权重失败: {e}")
        return

    print("📊 正在评估最佳模型...")
    try:
        heatmaps = matcher.predict_heatmap(validation_data)  # 假设此方法存在
        SarOptMatch_evaluation.print_results(validation_data, heatmaps)  # 打印详细结果
    except Exception as e:
        print(f"⚠️ 评估过程中热图预测或结果打印失败: {e}")
        import traceback
        traceback.print_exc()
        # 即使评估部分失败，也尝试继续进行特征提取和可视化

    # ✅ 根据参数决定是否提取特征图
    if extract_features: # ✅ 确认这里的条件判断
        print("🔥 正在计算特征图...")
        feature_maps_for_vis = None
        try:
            FEATURE_EXTRACTION_BATCH_SIZE = 4
            feature_maps_for_vis = matcher.calculate_features(
                validation_data,
                batch_size_for_feature_extraction=FEATURE_EXTRACTION_BATCH_SIZE
                # output_root_dir="evaluation_features" # 如果您的 calculate_features 需要这个
            )
            if feature_maps_for_vis:
                print(f"  ✅ 特征图已提取 (数量: {len(feature_maps_for_vis)}).")
            else:
                print(f"  ℹ️ 特征图提取未返回任何内容或被跳过。")
        except Exception as e:
            print(f"⚠️ 特征图计算失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("ℹ️ 根据设置，跳过特征图提取。") # ✅ 确认有这个 else 分支的打印



# =======================================================================================
# === 改动开始: plot_training_curve 函数 ===
# =======================================================================================
def plot_training_curve(model_name, history_object=None, score_history_list=None):
    """
    绘制训练/验证损失曲线，以及可选的评估分数曲线。

    Args:
        model_name (str): 模型的名称，用于文件名和标题。
        history_object (tf.keras.callbacks.History, optional): Keras训练历史对象。
                                                             如果提供，将从中提取loss和val_loss。
        score_history_list (list, optional): 包含每轮评估分数的列表。
                                            如果提供，将绘制在第二个Y轴上。
    """
    print(f"📊 开始为模型 '{model_name}' 绘制训练曲线...")

    loss = []
    val_loss = []
    scores_to_plot = []  # 用于绘图的分数列表

    history_data_loaded = False
    if history_object and hasattr(history_object, 'history'):
        history_dict = history_object.history
        loss = history_dict.get('loss', [])
        val_loss = history_dict.get('val_loss', [])
        # 尝试从 history 对象中获取 'score' (如果 BestModelSaver 将其添加到 logs 中)
        scores_to_plot = history_dict.get('score', [])
        if scores_to_plot:
            print(f"  从 Keras History 对象中获取到 {len(scores_to_plot)} 个 score 值。")
        history_data_loaded = True
    else:
        # 如果没有 history_object，尝试从文件加载
        history_path = f"weights/{model_name}_history"  # 假设这是 Keras History 对象的 pickle 文件
        if os.path.exists(history_path):
            try:
                with open(history_path, 'rb') as f:
                    loaded_history_dict = pickle.load(f)  # 假设文件存的是字典或History对象

                if isinstance(loaded_history_dict, dict):
                    loss = loaded_history_dict.get('loss', [])
                    val_loss = loaded_history_dict.get('val_loss', [])
                    scores_to_plot = loaded_history_dict.get('score', [])  # 尝试获取 'score'
                    if scores_to_plot:
                        print(f"  从文件 '{history_path}' 的字典中获取到 {len(scores_to_plot)} 个 score 值。")
                elif hasattr(loaded_history_dict, 'history'):  # 如果加载的是 History 对象
                    loss = loaded_history_dict.history.get('loss', [])
                    val_loss = loaded_history_dict.history.get('val_loss', [])
                    scores_to_plot = loaded_history_dict.history.get('score', [])
                    if scores_to_plot:
                        print(f"  从文件 '{history_path}' 的 History 对象中获取到 {len(scores_to_plot)} 个 score 值。")
                else:
                    print(f"⚠️ 文件 '{history_path}' 的内容格式未知。")
                history_data_loaded = True
            except Exception as e:
                print(f"⚠️ 从 '{history_path}' 加载历史记录失败: {e}")
        else:
            print(f"⚠️ 未找到历史文件: {history_path} 且未提供 history_object。")

    # 如果从文件加载的历史记录中没有 'score'，但提供了 score_history_list，则使用它
    if not scores_to_plot and score_history_list is not None:
        scores_to_plot = score_history_list
        print(f"  使用通过参数传递的 {len(scores_to_plot)} 个 score 值。")
    elif not scores_to_plot:  # 如果到处都找不到score
        print("  ⚠️ 未找到或提供 score 数据进行绘图。")

    if not loss or not val_loss:
        if history_data_loaded:
            print("  历史记录中 loss 或 val_loss 为空，无法绘制损失曲线。")
        else:
            print("  未能加载任何历史数据 (loss/val_loss)，无法绘制损失曲线。")
        # 即使没有损失，如果只有分数，也可以尝试只绘制分数，但通常一起绘制才有意义
        if not scores_to_plot:
            return  # 如果什么数据都没有，直接返回

    # 确保所有列表长度一致，或者取最短的长度作为epoch范围
    num_epochs_loss = len(loss)
    num_epochs_val_loss = len(val_loss)
    num_epochs_score = len(scores_to_plot)

    # 以loss的长度为准，如果其他列表更短，绘图时会出问题
    # 通常它们应该等长，代表每个epoch的数据
    if num_epochs_loss == 0 and num_epochs_score > 0:  # 只有score
        epochs_range = range(1, num_epochs_score + 1)
        print("  仅绘制 Score 曲线。")
    elif num_epochs_loss > 0:
        epochs_range = range(1, num_epochs_loss + 1)
        if num_epochs_val_loss != num_epochs_loss and num_epochs_val_loss > 0:
            print(
                f"  警告: val_loss 长度 ({num_epochs_val_loss}) 与 loss 长度 ({num_epochs_loss}) 不匹配。将按 loss 长度截断/扩展。")
            # 简单处理：截断或用NaN填充，但最好是确保它们长度一致
        if scores_to_plot and num_epochs_score != num_epochs_loss and num_epochs_score > 0:
            print(
                f"  警告: score 长度 ({num_epochs_score}) 与 loss 长度 ({num_epochs_loss}) 不匹配。将按 loss 长度截断/扩展。")
    else:
        print("  没有有效的 epoch 数据进行绘图。")
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))  # 稍微调大尺寸

    # 左轴：损失
    if loss:
        ax1.plot(epochs_range[:len(loss)], loss, label='Train Loss', color='tab:blue', marker='o', linestyle='-')
    if val_loss:
        ax1.plot(epochs_range[:len(val_loss)], val_loss, label='Val Loss', color='tab:orange', marker='^',
                 linestyle='--')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue' if loss or val_loss else 'black')
    if loss or val_loss:
        ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 右轴：得分
    if scores_to_plot:
        ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴
        # 确保 scores_to_plot 的长度与 epochs_range 匹配
        ax2.plot(epochs_range[:len(scores_to_plot)], scores_to_plot, label='Evaluation Score', color='tab:green',
                 marker='s', linestyle='-.')
        ax2.set_ylabel('Evaluation Score', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.legend(loc='upper right')

    # 添加时间戳和标题
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.title(f'{model_name} | Training Curve ({timestamp})', fontsize=14)
    fig.tight_layout()  # 调整布局以防止标签重叠

    # 文件名也加入时间戳
    os.makedirs("weights", exist_ok=True)  # 确保目录存在
    save_path = f"weights/{model_name}_curve_{timestamp}.png"
    try:
        plt.savefig(save_path)
        plt.close(fig)  # 关闭图形，释放内存
        print(f"✅ 曲线图已保存到: {save_path}")
    except Exception as e:
        print(f"⚠️ 保存曲线图失败: {e}")
        plt.close(fig)  # 即使保存失败也尝试关闭

# =======================================================================================
# === 改动结束 ===
# =======================================================================================

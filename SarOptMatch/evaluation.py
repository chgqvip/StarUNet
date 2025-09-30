import glob
import os

import numpy as np
import matplotlib.pyplot as plt

# import SarOptMatch # 如果这个文件在 SarOptMatch 包内，则不需要


# SarOptMatch/evaluation.py
def get_predictions(data, heatmaps, batch_size=4):  # data 是 validation_data
    preds = []
    iter_val = iter(data)
    # 确保 heatmaps 的迭代次数与 data 中的批次数量匹配
    # 如果 heatmaps 是 (total_samples, H, W) 或 (total_samples, H, W, C)
    # 并且我们想按批次处理，需要确保 heatmaps 的第一个维度是 num_batches
    # 从 print_results 看，heatmaps_reshaped 已经是 (num_batches, bs, H, W, C)
    num_batches = len(heatmaps) # heatmaps 已经是按批次组织的

    for i in range(num_batches):
        try:
            _inputs_batch, targets_tuple_batch = iter_val.get_next()
        except tf.errors.OutOfRangeError: # 捕捉迭代器耗尽的情况
            print(f"Warning: Validation data iterator exhausted at batch {i+1}/{num_batches}. "
                  f"Heatmaps may not be fully processed.")
            break # 如果数据提前耗尽，则停止处理

        true_mask_batch = targets_tuple_batch[0]

        # 确保当前批次的 heatmaps 维度与 batch_size 一致
        current_batch_heatmaps = heatmaps[i]
        actual_batch_size = current_batch_heatmaps.shape[0] # 当前批次实际大小
        if actual_batch_size != true_mask_batch.shape[0]:
            print(f"Warning: Mismatch in batch size between heatmaps ({actual_batch_size}) "
                  f"and true masks ({true_mask_batch.shape[0]}) at batch {i}.")
            # 可以选择跳过这个批次或尝试处理匹配的部分
            continue


        for j in range(actual_batch_size): # 使用当前批次的实际大小
            true_peak_coords = np.unravel_index(true_mask_batch[j].numpy().argmax(), true_mask_batch[j].numpy().shape)

            # current_batch_heatmaps[j] 是单个样本的预测热图
            # 确保 current_batch_heatmaps[j] 是 2D 的 (H,W)
            single_heatmap = current_batch_heatmaps[j]
            if single_heatmap.ndim > 2 and single_heatmap.shape[-1] == 1: # (H, W, 1) -> (H, W)
                single_heatmap = np.squeeze(single_heatmap, axis=-1)
            elif single_heatmap.ndim != 2:
                print(f"Warning: Expected 2D or (H,W,1) heatmap, got shape {single_heatmap.shape} for sample {j} in batch {i}. Skipping.")
                preds.append(((0,0), (0,0))) # 添加占位符或跳过
                continue


            pred_peak_coords_flat = single_heatmap.argmax()
            pred_peak_coords = np.unravel_index(pred_peak_coords_flat, single_heatmap.shape)

            preds.append((true_peak_coords, pred_peak_coords))
    return preds


# === 修改：增加 dx_errors 和 dy_errors 参数 ===
def print_perf_table(euc_dist, dx_errors, dy_errors):
    print("Matching Accuracy")
    print("<= 1 pixel: " +str(np.where(euc_dist <= 1, 1,0).sum()/len(euc_dist)) )
    print("<= 2 pixel: " +str(np.where(euc_dist <= 2, 1,0).sum()/len(euc_dist)) )
    print("<= 3 pixel: " +str(np.where(euc_dist <= 3, 1,0).sum()/len(euc_dist)) )
    print("<= 5 pixel: " +str(np.where(euc_dist <= 5, 1,0).sum()/len(euc_dist)) )

    print(f"Average Euclidean Distance: {euc_dist.mean():.4f}")

    # === 新增：打印 X 和 Y 方向的偏移统计 ===
    if dx_errors.size > 0 and dy_errors.size > 0: # 确保数组不为空
        print(f"Average X Offset (pred - true): {dx_errors.mean():.4f} pixels")
        print(f"Average Y Offset (pred - true): {dy_errors.mean():.4f} pixels")
        print(f"Average Absolute X Offset: {np.abs(dx_errors).mean():.4f} pixels")
        print(f"Average Absolute Y Offset: {np.abs(dy_errors).mean():.4f} pixels")
    else:
        print("Warning: No X/Y offset data to report.")
    # === 新增结束 ===

def print_results(validation_data, heatmaps):
    # 假设 validation_data 是一个 tf.data.Dataset 对象
    # 我们需要知道数据集中的样本总数来正确 reshape heatmaps
    # 或者，如果 heatmaps 的第一个维度已经是批次数，我们可以直接用
    # 假设 heatmaps 的形状是 (total_samples, H, W, C) 或 (total_samples, H, W)
    # 而 validation_data 的 batch_size 是 bs

    # 尝试从 validation_data 获取 batch_size
    # 这通常在创建 dataset 时指定，或者可以从元素规范推断，但直接用已知值更简单
    # 如果 validation_data 是 unbatch().batch(bs) 创建的，bs 是已知的
    # 这里我们假设 bs 是已知的，例如传入或者从全局获取
    bs = 4 # 假设批大小是4，与 get_predictions 中的默认值一致
           # 如果 validation_data 的批大小不同，这里需要调整

    num_samples = heatmaps.shape[0]
    if num_samples == 0:
        print("Warning: No heatmaps provided to print_results.")
        return np.array([])

    if num_samples % bs != 0:
        print(f"Warning: Total number of heatmap samples ({num_samples}) is not a multiple of batch size ({bs}). "
              f"Reshaping might be incorrect or some samples might be dropped by reshape.")
        # 可以考虑只处理能被 bs 整除的部分
        num_batches_full = num_samples // bs
        heatmaps_to_reshape = heatmaps[:num_batches_full * bs]
        if heatmaps_to_reshape.size == 0 and num_samples > 0:
             print("Warning: Not enough samples for even one full batch after considering batch size. Cannot proceed with reshape.")
             return np.array([])
        elif heatmaps_to_reshape.size == 0 and num_samples == 0:
             return np.array([])

    else:
        heatmaps_to_reshape = heatmaps
        num_batches_full = num_samples // bs


    # 确保热图至少有3个维度 (num_samples, H, W) 才能 reshape
    if heatmaps_to_reshape.ndim < 3:
        print(f"Warning: Heatmaps have insufficient dimensions ({heatmaps_to_reshape.ndim}) for reshaping. Expected at least 3.")
        return np.array([])

    # Reshape heatmaps to (num_batches, bs, H, W, C) or (num_batches, bs, H, W)
    # 如果原始 heatmaps 是 (total_samples, H, W)，则 reshape 为 (num_batches, bs, H, W)
    # 如果原始 heatmaps 是 (total_samples, H, W, C)，则 reshape 为 (num_batches, bs, H, W, C)
    h_dim = heatmaps_to_reshape.shape[1]
    w_dim = heatmaps_to_reshape.shape[2]
    if heatmaps_to_reshape.ndim == 4: # (total_samples, H, W, C)
        c_dim = heatmaps_to_reshape.shape[3]
        try:
            heatmaps_reshaped = heatmaps_to_reshape.reshape([num_batches_full, bs, h_dim, w_dim, c_dim])
        except ValueError as e:
            print(f"Error reshaping heatmaps with 4 dims: {e}. Shape was {heatmaps_to_reshape.shape}, target batch size {bs}")
            return np.array([])
    elif heatmaps_to_reshape.ndim == 3: # (total_samples, H, W)
        try:
            heatmaps_reshaped = heatmaps_to_reshape.reshape([num_batches_full, bs, h_dim, w_dim])
        except ValueError as e:
            print(f"Error reshaping heatmaps with 3 dims: {e}. Shape was {heatmaps_to_reshape.shape}, target batch size {bs}")
            return np.array([])
    else:
        print(f"Warning: Unexpected heatmap dimension {heatmaps_to_reshape.ndim}. Cannot reshape correctly.")
        return np.array([])


    result = get_predictions(validation_data, heatmaps_reshaped, bs) # 将 bs 传递给 get_predictions

    if not result: # 如果 get_predictions 返回空列表
        print("Warning: get_predictions returned no results.")
        return np.array([])

    true_coords, predicted_coords = zip(*result)
    true_coords_arr = np.array(true_coords)     # Shape: (N, 2) -> (y_true, x_true)
    predicted_coords_arr = np.array(predicted_coords) # Shape: (N, 2) -> (y_pred, x_pred)

    if true_coords_arr.size == 0 or predicted_coords_arr.size == 0:
        print("Warning: No valid true or predicted coordinates obtained.")
        return np.array([])


    euc_dists = np.linalg.norm(true_coords_arr - predicted_coords_arr, axis=1)

    # === 新增：计算 X 和 Y 方向的偏移 ===
    # true_coords_arr[:, 0] 是所有 y_true, true_coords_arr[:, 1] 是所有 x_true
    # predicted_coords_arr[:, 0] 是所有 y_pred, predicted_coords_arr[:, 1] 是所有 x_pred
    dy_errors = predicted_coords_arr[:, 0] - true_coords_arr[:, 0] # y_pred - y_true
    dx_errors = predicted_coords_arr[:, 1] - true_coords_arr[:, 1] # x_pred - x_true
    # === 新增结束 ===

    # === 修改：将 dx_errors 和 dy_errors 传递给 print_perf_table ===
    print_perf_table(euc_dists, dx_errors, dy_errors)

    return euc_dists

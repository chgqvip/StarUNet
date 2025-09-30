# SarOptMatch/evaluate_with_confidence.py
import traceback

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import os
import cv2  # <--- 确保导入 OpenCV

try:
    # 尝试相对路径导入
    from SarOptMatch.evaluation import print_perf_table
except ImportError as e_main:
    print(f"警告: 无法通过相对路径导入 SarOptMatch.evaluation: {e_main}。尝试直接导入。")
    try:
        # 如果模块结构是 SarOptMatch.evaluation，尝试直接导入
        import SarOptMatch.evaluation as evaluation_module

        print_perf_table = evaluation_module.print_perf_table
        print("成功通过直接导入 SarOptMatch.evaluation 导入 print_perf_table。")
    except ImportError as e_direct:
        print(f"错误: 从 SarOptMatch.evaluation 导入 print_perf_table 失败: {e_direct}。")


        # 占位符函数，确保它不使用 group_name
        def print_perf_table(euc_dists, dx_errors=None, dy_errors=None):  # 移除了 **kwargs 以匹配错误
            print("--- 占位符 print_perf_table ---")
            if euc_dists is not None and euc_dists.size > 0:
                print(f"  平均欧氏距离: {np.mean(euc_dists):.4f}")
            else:
                print("  欧氏距离: 无")

            if dx_errors is not None and dx_errors.size > 0:
                print(f"  平均X方向偏移 (预测 - 真实): {np.mean(dx_errors):.4f} 像素")
                print(f"  平均X方向绝对偏移: {np.mean(np.abs(dx_errors)):.4f} 像素")
            else:
                print("  dx_errors: 无")

            if dy_errors is not None and dy_errors.size > 0:
                print(f"  平均Y方向偏移 (预测 - 真实): {np.mean(dy_errors):.4f} 像素")
                print(f"  平均Y方向绝对偏移: {np.mean(np.abs(dy_errors)):.4f} 像素")
            else:
                print("  dy_errors: 无")
            print("--- 占位符结束 ---")


def calculate_entropy(heatmap_probs):
    """计算单个热图的熵值 (基于概率分布)"""
    heatmap_probs = np.maximum(heatmap_probs, 1e-12)  # 避免 log(0) 错误
    entropy = -np.sum(heatmap_probs * np.log2(heatmap_probs))
    return entropy


def spatial_softmax_numpy(raw_heatmap):
    """对单个原始热图应用空间Softmax (NumPy版本)，转换为概率分布"""
    flat_heatmap = raw_heatmap.flatten()
    # 数值稳定性：减去最大值防止溢出
    exp_heatmap = np.exp(flat_heatmap - np.max(flat_heatmap))
    softmax_flat = exp_heatmap / np.sum(exp_heatmap)
    return softmax_flat.reshape(raw_heatmap.shape)


def run_confidence_evaluation_with_entropy(
        model_to_evaluate: tf.keras.Model,
        validation_data_for_confidence: tf.data.Dataset,
        model_name_for_log: str,
        entropy_percentile_for_selection=20,  # 用于划分低熵/高熵组的百分位数
        log_highest_entropy_samples_percentile=80,  # 用于在直方图上标记的熵百分位数
        # === RANSAC 相关参数 (从主调用脚本传入) ===
        apply_ransac_on_low_entropy=False,  # 是否在低熵组上应用RANSAC
        ransac_reproj_threshold=5.0,  # RANSAC 重投影阈值 (像素)
        min_ransac_samples=4,  # RANSAC 所需最小样本数
        ##### 新增/修改 开始 #####
        run_ransac_only_on_all_samples_mode=False,  # 新增参数，控制是否只运行RANSAC模式
        ##### 新增/修改 结束 #####
        output_dir="evaluation_results"  # 评估结果输出目录
):
    M_affine_all = None  # 为 RANSAC-only 模式下的全局仿射矩阵初始化
    M_affine = None  # 为熵模式下低熵组的仿射矩阵初始化

    if run_ransac_only_on_all_samples_mode:
        print(f"\n--- 🛡️ 开始对模型 '{model_name_for_log}' 进行仅RANSAC评估 (在所有样本上) ---")
        print(f"RANSAC参数: 重投影阈值: {ransac_reproj_threshold}像素, 最少样本数: {min_ransac_samples}")
    else:
        print(f"\n--- 🛡️ 开始对模型 '{model_name_for_log}' 进行基于熵值的评估 ---")
        print(f"低熵组定义: 熵值最低的 {entropy_percentile_for_selection}% 的样本")
        if apply_ransac_on_low_entropy:  # 此参数仅在熵模式下有意义
            print(
                f"RANSAC评估 (熵模式): 将在低熵组上应用RANSAC (重投影阈值: {ransac_reproj_threshold}像素, 最少样本数: {min_ransac_samples})")


    # 为每个模型创建一个子目录，替换掉文件名中不合法字符
    model_output_dir = os.path.join(output_dir,
                                    model_name_for_log.replace(' ', '_').replace('/', '_').replace(':', '_'))
    os.makedirs(model_output_dir, exist_ok=True)

    # =================================================================
    # vvvvvvvvvvvvvvvv 【核心修改：分步处理以避免OOM】 vvvvvvvvvvvvvvvv
    # =================================================================

    # --- 步骤 1: 分批预测与数据收集 ---
    all_raw_heatmaps_list = []
    all_gt_offsets_list = []

    desc_str = f"模型预测 ({model_name_for_log})"
    progress_bar = tqdm(validation_data_for_confidence, desc=desc_str, leave=False, file=sys.stdout, unit="batch")

    for batch_idx, batch_data in enumerate(progress_bar):
        try:
            # 兼容不同数据集结构
            if len(batch_data) == 2 and isinstance(batch_data[0], tuple) and isinstance(batch_data[1], tuple):
                (opt_im_batch, sar_im_batch), (_, gt_original_offsets_batch) = batch_data
            else:
                opt_im_batch, sar_im_batch, _, gt_original_offsets_batch = batch_data

            raw_heatmaps_batch = model_to_evaluate.predict_on_batch([opt_im_batch, sar_im_batch])

            all_raw_heatmaps_list.append(raw_heatmaps_batch)
            all_gt_offsets_list.append(gt_original_offsets_batch)

        except Exception as e:
            print(f"\n❌ 错误: 在批次 {batch_idx} 上进行模型预测失败: {e}")
            print(f"   光学图像批次形状: {opt_im_batch.shape}, SAR图像批次形状: {sar_im_batch.shape}")
            traceback.print_exc()
            continue

    if not all_raw_heatmaps_list:
        print("未处理任何有效样本。评估中止。")
        return [], None

    # --- 步骤 2: 汇总所有批次的结果 ---
    print(f"\n...所有批次预测完成，正在汇总 {len(all_raw_heatmaps_list)} 个批次的结果...")
    # 使用 .numpy() 确保从Tensor转换为Numpy数组后再拼接
    full_raw_heatmaps = np.concatenate([h for h in all_raw_heatmaps_list], axis=0)
    full_gt_offsets = np.concatenate([o.numpy() if hasattr(o, 'numpy') else o for o in all_gt_offsets_list], axis=0)
    print(f"   成功汇总了 {full_raw_heatmaps.shape[0]} 个样本的预测结果。")

    # --- 步骤 3: 使用汇总后的数据进行后续所有分析 ---
    sample_metrics_data = []
    for i in range(full_raw_heatmaps.shape[0]):
        raw_heatmap_sample = full_raw_heatmaps[i, :, :, 0]
        gt_offset_numpy = full_gt_offsets[i]

        gt_offset_y, gt_offset_x = gt_offset_numpy
        pred_offset_y_final, pred_offset_x_final = np.unravel_index(np.argmax(raw_heatmap_sample),
                                                                    raw_heatmap_sample.shape)

        gt_np_coords = np.array([gt_offset_y, gt_offset_x])
        pred_np_coords = np.array([pred_offset_y_final, pred_offset_x_final])

        euc_dist = np.linalg.norm(gt_np_coords - pred_np_coords)
        dy_error = pred_offset_y_final - gt_offset_y
        dx_error = pred_offset_x_final - gt_offset_x

        current_sample_dict = {
            'gt_offset': (gt_offset_y, gt_offset_x),
            'pred_offset': (pred_offset_y_final, pred_offset_x_final),
            'euc_dist': euc_dist,
            'dx_error': dx_error,
            'dy_error': dy_error,
            'index': i
        }

        if not run_ransac_only_on_all_samples_mode:
            prob_heatmap_sample = spatial_softmax_numpy(raw_heatmap_sample)
            entropy_value = calculate_entropy(prob_heatmap_sample)
            current_sample_dict['entropy'] = entropy_value

        sample_metrics_data.append(current_sample_dict)

    # =================================================================
    # ^^^^^^^^^^^^^^^^ 【核心修改结束】 ^^^^^^^^^^^^^^^^


    if not sample_metrics_data:
        print("未处理任何有效样本。评估中止。")
        return [], None  # 返回空列表和None

    num_total_samples = len(sample_metrics_data)  # 总样本数对两种模式都有用
    print(f"\n总有效样本数: {num_total_samples}")  # 提前打印总样本数

    ##### 新增/修改 开始 #####
    if run_ransac_only_on_all_samples_mode:
        # --- 仅 RANSAC 评估 (在所有样本上) ---
        if num_total_samples >= min_ransac_samples:
            print(f"\n  --- RANSAC 评估 (在全部 {num_total_samples} 个样本上) ---")
            # OpenCV RANSAC 需要点坐标格式为 (x, y)
            # src_pts: 模型的预测偏移 (pred_offset)，作为源点
            # dst_pts: 真实的偏移 (gt_offset)，作为目标点
            # RANSAC的目标是找到一个变换 M，使得 M(src_pts) ≈ dst_pts
            src_pts_list_all = [[item['pred_offset'][1], item['pred_offset'][0]] for item in
                                sample_metrics_data]  # (x_pred, y_pred)
            dst_pts_list_all = [[item['gt_offset'][1], item['gt_offset'][0]] for item in
                                sample_metrics_data]  # (x_gt, y_gt)

            src_pts_np_all = np.float32(src_pts_list_all).reshape(-1, 1, 2)
            dst_pts_np_all = np.float32(dst_pts_list_all).reshape(-1, 1, 2)

            # M_affine_all 在函数开始时已初始化为 None
            try:
                # 使用OpenCV的RANSAC估计仿射变换矩阵
                # M_affine_all 将 src_pts_np_all (预测偏移) 变换到接近 dst_pts_np_all (真实偏移)
                M_affine_all, inliers_mask_affine_all = cv2.estimateAffine2D(
                    src_pts_np_all, dst_pts_np_all,  # 源点: 预测偏移, 目标点: 真实偏移
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_reproj_threshold,  # 重投影阈值
                    maxIters=2000, confidence=0.99  # 最大迭代次数和置信度
                )
                if M_affine_all is not None:
                    num_inliers_all = np.sum(inliers_mask_affine_all)  # 内点数量
                    inlier_ratio_all = num_inliers_all / num_total_samples if num_total_samples > 0 else 0  # 内点比例
                    print(f"      RANSAC (仿射变换) 结果:")
                    print(f"        总点数 (全部样本): {num_total_samples}")
                    print(f"        内点数: {num_inliers_all}")
                    print(f"        内点比例: {inlier_ratio_all:.4f}")
                    print(f"        估计的仿射变换矩阵 M (将预测偏移变换至接近真实偏移):\n{M_affine_all}")

                    if num_inliers_all > 0:
                        # 提取内点
                        src_inliers_all = src_pts_np_all[inliers_mask_affine_all.ravel() == 1]
                        dst_inliers_all = dst_pts_np_all[inliers_mask_affine_all.ravel() == 1]
                        # 对内点源点应用估计的变换
                        transformed_src_inliers_raw_all = cv2.transform(src_inliers_all, M_affine_all)
                        transformed_src_inliers_flat_all = transformed_src_inliers_raw_all.reshape(-1, 2)
                        dst_inliers_flat_all = dst_inliers_all.reshape(-1, 2)
                        # 计算内点的重投影误差
                        reprojection_errors_inliers_all = np.linalg.norm(
                            transformed_src_inliers_flat_all - dst_inliers_flat_all, axis=1)
                        print(f"        内点平均重投影误差: {np.mean(reprojection_errors_inliers_all):.4f} 像素")
                        print(f"        内点中位重投影误差: {np.median(reprojection_errors_inliers_all):.4f} 像素")
                        # 计算内点的RMSE
                        rmse_inliers = np.sqrt(np.mean(np.square(reprojection_errors_inliers_all)))
                        print(f"        内点RMSE (M(预测偏移) vs 真实偏移): {rmse_inliers:.4f} 像素")

                    ##### 最小侵入式RMSE计算 (针对所有点应用全局变换) #####
                    print(f"\n      --- 全局RMSE计算 (所有样本应用M_affine_all) ---")
                    try:
                        # 使用估计的仿射模型 M_affine_all 变换所有的源点 (即所有预测的局部偏移)
                        transformed_all_pred_offsets_raw = cv2.transform(src_pts_np_all, M_affine_all)
                        transformed_all_pred_offsets_flat = transformed_all_pred_offsets_raw.reshape(-1, 2)  # (N, 2)

                        # 真实的局部偏移 (目标点)，扁平化
                        all_gt_offsets_flat = dst_pts_np_all.reshape(-1, 2)  # (N, 2)

                        # 计算应用RANSAC模型后，所有变换后的预测偏移与它们对应真实偏移之间的欧氏距离误差
                        # errors_all_points_after_affine 是一个包含每个点对误差的数组
                        errors_all_points_after_affine = np.linalg.norm(
                            transformed_all_pred_offsets_flat - all_gt_offsets_flat, axis=1)

                        # RMSE = sqrt(mean(squared_errors))
                        rmse_all_points_after_affine_correction = np.sqrt(
                            np.mean(np.square(errors_all_points_after_affine)))
                        print(
                            f"        全局RMSE (所有 {num_total_samples} 个样本, M_affine_all(预测偏移) vs 真实偏移): {rmse_all_points_after_affine_correction:.4f} 像素")
                    except Exception as e_rmse_calc:
                        print(f"        错误: 计算全局RMSE失败: {e_rmse_calc}")
                    ##### 最小侵入式RMSE计算结束 #####

                else:
                    print(f"      RANSAC (仿射变换) 未能估计变换模型 (在全部样本上)。M_affine_all 为 None。")
            except cv2.error as e_cv2_all:  # OpenCV特定错误
                print(f"      RANSAC (仿射变换) 执行失败 (全部样本, cv2.error): {e_cv2_all}")
                if M_affine_all is not None:  # 如果在transform中出错，M可能已赋值
                    print(f"        错误发生时 M_affine_all 的值: {M_affine_all}")
            except Exception as e_ransac_all:  # 其他通用错误
                print(f"      RANSAC (仿射变换) 评估中发生未知错误 (全部样本): {e_ransac_all}")
        else:
            print(f"\n  --- RANSAC 评估 (在全部样本上) ---")
            print(f"      样本数 ({num_total_samples}) 少于 RANSAC 所需最小样本数 ({min_ransac_samples})，跳过RANSAC。")

        # 可选：打印所有样本的原始误差统计 (未经RANSAC校正的)
        print("\n  --- 所有样本的原始误差统计 (仅RANSAC模式, 未经RANSAC校正) ---")
        all_euc_dists = np.array([item['euc_dist'] for item in sample_metrics_data])
        all_dx_errors = np.array([item['dx_error'] for item in sample_metrics_data])
        all_dy_errors = np.array([item['dy_error'] for item in sample_metrics_data])
        if all_euc_dists.size > 0:
            print_perf_table(all_euc_dists, all_dx_errors, all_dy_errors)
        else:
            print("    没有样本数据可供统计。")

    else:  # 这是原始的基于熵的评估逻辑
        ##### 新增/修改 结束 #####
        # 确保 sample_metrics_data 中的每个 item 都有 'entropy'键 (在前面已处理)
        sorted_samples_by_entropy = sorted(sample_metrics_data, key=lambda x: x['entropy'])
        # num_total_samples 已经在前面从 sample_metrics_data 计算得出

        # 确定低熵组的样本数量
        if entropy_percentile_for_selection > 0:
            num_low_entropy_samples = int(np.ceil(num_total_samples * (entropy_percentile_for_selection / 100.0)))
            if num_low_entropy_samples == 0 and num_total_samples > 0:  # 至少选一个，如果百分比太小导致为0但有样本
                num_low_entropy_samples = 1
        else:  # 如果百分比为0，则低熵组为空
            num_low_entropy_samples = 0
        num_low_entropy_samples = min(num_low_entropy_samples, num_total_samples)  # 不能超过总样本数

        low_entropy_group = sorted_samples_by_entropy[:num_low_entropy_samples]
        high_entropy_group = sorted_samples_by_entropy[num_low_entropy_samples:]

        print(f"\n--- 模型 '{model_name_for_log}' 分组评估结果 (熵模式) ---")
        # print(f"总有效样本数: {num_total_samples}") # 已提前打印

        # --- 低熵组评估 ---
        print(
            f"\n  --- 低熵组 (熵值最低的 {entropy_percentile_for_selection}%, 共 {len(low_entropy_group)} 个样本) ---")
        low_entropy_threshold_display = -1.0  # 用于显示的熵阈值
        if low_entropy_group:
            low_entropy_euc_dists = np.array([item['euc_dist'] for item in low_entropy_group])
            low_entropy_dx_errors = np.array([item['dx_error'] for item in low_entropy_group])
            low_entropy_dy_errors = np.array([item['dy_error'] for item in low_entropy_group])

            if low_entropy_euc_dists.size > 0:
                print_perf_table(low_entropy_euc_dists, low_entropy_dx_errors, low_entropy_dy_errors)

            low_entropy_threshold_display = low_entropy_group[-1]['entropy']  # 低熵组中最大的熵值
            print(f"    (样本熵值 <= {low_entropy_threshold_display:.4f})")

            # 打印低熵组中部分样本的详细信息 (首尾几个)
            num_to_show_low = min(len(low_entropy_group), 7)
            show_first_n_low = (num_to_show_low + 1) // 2
            show_last_n_low = num_to_show_low // 2
            for k, sample_detail in enumerate(low_entropy_group):
                if k < show_first_n_low or k >= len(low_entropy_group) - show_last_n_low:
                    print(
                        f"      样本 [索引:{sample_detail['index']}]: 熵={sample_detail['entropy']:.4f}, 欧氏距离={sample_detail['euc_dist']:.2f}, GT=({sample_detail['gt_offset'][0]:.1f},{sample_detail['gt_offset'][1]:.1f}), Pred=({sample_detail['pred_offset'][0]},{sample_detail['pred_offset'][1]}), dx误差={sample_detail['dx_error']:.2f}, dy误差={sample_detail['dy_error']:.2f}")
                elif k == show_first_n_low and len(low_entropy_group) > num_to_show_low:
                    print("      ...")  # 省略中间的样本

            if apply_ransac_on_low_entropy and len(low_entropy_group) >= min_ransac_samples:
                print(f"\n    --- RANSAC 评估 (低熵组) ---")
                src_pts_list = [[item['pred_offset'][1], item['pred_offset'][0]] for item in low_entropy_group]
                dst_pts_list = [[item['gt_offset'][1], item['gt_offset'][0]] for item in low_entropy_group]
                src_pts_np = np.float32(src_pts_list).reshape(-1, 1, 2)
                dst_pts_np = np.float32(dst_pts_list).reshape(-1, 1, 2)

                # M_affine 在函数开始时已初始化为 None
                try:
                    M_affine, inliers_mask_affine = cv2.estimateAffine2D(src_pts_np, dst_pts_np,
                                                                         method=cv2.RANSAC,
                                                                         ransacReprojThreshold=ransac_reproj_threshold,
                                                                         maxIters=2000, confidence=0.99)
                    if M_affine is not None:
                        num_inliers = np.sum(inliers_mask_affine)
                        inlier_ratio = num_inliers / len(low_entropy_group) if len(low_entropy_group) > 0 else 0
                        print(f"      RANSAC (仿射变换) 结果:")
                        print(f"        总点数 (低熵组): {len(low_entropy_group)}")
                        print(f"        内点数: {num_inliers}")
                        print(f"        内点比例: {inlier_ratio:.4f}")
                        print(f"        估计的仿射变换矩阵 M (将预测偏移变换至接近真实偏移):\n{M_affine}")
                        if num_inliers > 0:
                            src_inliers = src_pts_np[inliers_mask_affine.ravel() == 1]
                            dst_inliers = dst_pts_np[inliers_mask_affine.ravel() == 1]
                            transformed_src_inliers_raw = cv2.transform(src_inliers, M_affine)
                            transformed_src_inliers_flat = transformed_src_inliers_raw.reshape(-1, 2)
                            dst_inliers_flat = dst_inliers.reshape(-1, 2)
                            reprojection_errors_inliers = np.linalg.norm(
                                transformed_src_inliers_flat - dst_inliers_flat, axis=1)
                            print(f"        内点平均重投影误差: {np.mean(reprojection_errors_inliers):.4f} 像素")
                            print(f"        内点中位重投影误差: {np.median(reprojection_errors_inliers):.4f} 像素")
                            rmse_inliers_low_entropy = np.sqrt(np.mean(np.square(reprojection_errors_inliers)))
                            print(f"        内点RMSE (M(预测偏移) vs 真实偏移): {rmse_inliers_low_entropy:.4f} 像素")
                    else:
                        print(f"      RANSAC (仿射变换) 未能估计变换模型 (低熵组, 可能内点不足或点共线等)。")
                except cv2.error as e_cv2:
                    print(f"      RANSAC (仿射变换) 执行失败 (低熵组, cv2.error): {e_cv2}")
                    if M_affine is not None:
                        print(f"        错误发生时 M_affine 的值: {M_affine}")
                except Exception as e_ransac:
                    print(f"      RANSAC (仿射变换) 评估中发生未知错误 (低熵组): {e_ransac}")
            elif apply_ransac_on_low_entropy:  # 如果启用了RANSAC但样本不足
                print(f"\n    --- RANSAC 评估 (低熵组) ---")
                print(
                    f"      样本数 ({len(low_entropy_group)}) 少于 RANSAC 所需最小样本数 ({min_ransac_samples})，跳过RANSAC。")
        else:
            print("    低熵组中没有样本。")

        # --- 高熵组评估 ---
        print(f"\n  --- 高熵组 (剩余 {len(high_entropy_group)} 个样本) ---")
        if high_entropy_group:
            high_entropy_euc_dists = np.array([item['euc_dist'] for item in high_entropy_group])
            high_entropy_dx_errors = np.array([item['dx_error'] for item in high_entropy_group])
            high_entropy_dy_errors = np.array([item['dy_error'] for item in high_entropy_group])
            if high_entropy_euc_dists.size > 0:
                print_perf_table(high_entropy_euc_dists, high_entropy_dx_errors, high_entropy_dy_errors)

            actual_high_entropy_start_threshold = high_entropy_group[0]['entropy'] if high_entropy_group else float(
                'inf')
            if low_entropy_threshold_display != -1.0:  # 如果低熵组存在且有阈值
                print(
                    f"    (样本熵值 > {low_entropy_threshold_display:.4f}, 实际起始于 {actual_high_entropy_start_threshold:.4f})")
            else:  # 如果所有样本都在高熵组
                print(f"    (所有样本均在此组, 实际起始熵值: {actual_high_entropy_start_threshold:.4f})")

            # 打印高熵组中部分样本的详细信息
            num_to_show_high = min(len(high_entropy_group), 7)
            show_first_n_high = (num_to_show_high + 1) // 2
            show_last_n_high = num_to_show_high // 2
            for k, sample_detail in enumerate(high_entropy_group):
                if k < show_first_n_high or k >= len(high_entropy_group) - show_last_n_high:
                    print(
                        f"      样本 [索引:{sample_detail['index']}]: 熵={sample_detail['entropy']:.4f}, 欧氏距离={sample_detail['euc_dist']:.2f}, GT=({sample_detail['gt_offset'][0]:.1f},{sample_detail['gt_offset'][1]:.1f}), Pred=({sample_detail['pred_offset'][0]},{sample_detail['pred_offset'][1]}), dx误差={sample_detail['dx_error']:.2f}, dy误差={sample_detail['dy_error']:.2f}")
                elif k == show_first_n_high and len(high_entropy_group) > num_to_show_high:
                    print("      ...")
        else:
            print("    高熵组中没有样本。")

        # --- 绘制直方图 (仅在熵模式下) ---
        all_entropies_np = np.array([item['entropy'] for item in sorted_samples_by_entropy])
        if all_entropies_np.size == 0:
            print("没有有效的熵值数据用于绘制直方图。")
        else:
            print("\n📊 开始绘制熵值直方图...")
            plt.figure(figsize=(12, 7))
            plt.hist(all_entropies_np, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
            plt.title(f'{model_name_for_log} 的热图熵值分布\n总样本数: {num_total_samples}', fontsize=14)
            plt.xlabel('熵 (比特)', fontsize=12)
            plt.ylabel('频率', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # 标记低熵/高熵分割线
            if low_entropy_group and high_entropy_group:  # 只有当两组都存在时，分割线才有意义
                threshold_value_for_plot = low_entropy_group[-1]['entropy']
                plt.axvline(threshold_value_for_plot, color='dodgerblue', linestyle='dashed', linewidth=2,
                            label=f'低/高熵阈值 (最低 {entropy_percentile_for_selection}%)\n熵 $\leq$ {threshold_value_for_plot:.2f}')
            elif low_entropy_group and not high_entropy_group:  # 所有样本都在低熵组
                plt.text(0.98, 0.95, '所有样本均在低熵组', ha='right', va='top',
                         transform=plt.gca().transAxes, color='dodgerblue')
            elif not low_entropy_group and high_entropy_group:  # 所有样本都在高熵组
                plt.text(0.98, 0.95, '所有样本均在高熵组', ha='right', va='top',
                         transform=plt.gca().transAxes, color='darkorange')

            # 标记用于日志记录的高熵百分位点
            if 0 < log_highest_entropy_samples_percentile < 100:
                log_percentile_value_for_plot = np.percentile(all_entropies_np, log_highest_entropy_samples_percentile)
                plt.axvline(log_percentile_value_for_plot, color='darkviolet', linestyle='dotted', linewidth=2,
                            label=f'{log_highest_entropy_samples_percentile}百分位熵\n值 $\geq$ {log_percentile_value_for_plot:.2f}')

            if plt.gca().has_data():  # 确保图例有内容可显示
                plt.legend(fontsize='small', loc='best')
            plt.tight_layout()  # 调整布局以防止标签重叠
            plot_filename = os.path.join(model_output_dir, f"entropy_histogram_grouped.png")
            try:
                plt.savefig(plot_filename)
                print(f"✅ 熵值直方图已保存到: {plot_filename}")
            except Exception as e:
                print(f"⚠️ 无法保存熵值直方图: {e}")
            plt.close()  # 关闭图像以释放内存

    M_to_return = None  # 初始化要返回的变换矩阵
    if run_ransac_only_on_all_samples_mode:
        print(f"\n--- 🛡️ 模型 '{model_name_for_log}' 的仅RANSAC评估结束 ---")
        M_to_return = M_affine_all  # 在仅RANSAC模式下，返回在所有样本上计算的矩阵
    else:
        print(f"\n--- 🛡️ 模型 '{model_name_for_log}' 的基于熵值的评估结束 ---")
        if apply_ransac_on_low_entropy and M_affine is not None:  # 检查在熵模式下是否计算了M_affine
            M_to_return = M_affine  # 在熵模式下，如果应用了RANSAC并成功，则返回低熵组的矩阵
        # 否则 M_to_return 保持为 None

    return sample_metrics_data, M_to_return

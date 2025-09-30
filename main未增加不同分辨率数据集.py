# main.py
import os
import tensorflow as tf
import numpy as np
import random
import traceback
import gc
import sys
from tqdm import tqdm
import SarOptMatch  # 你的包
from SarOptMatch.train_modules import BestModelSaver, evaluate_best_model, plot_training_curve
from SarOptMatch.log import setup_logger
from tensorflow.keras import backend as K
import pickle
import time
import glob
import matplotlib.pyplot as plt
import matplotlib

# =================================================================
# <<< 新增代码：初始化时间和内存监控 >>>
# =================================================================
script_start_time = time.time()
psutil_available = False
try:
    import psutil

    process = psutil.Process(os.getpid())
    psutil_available = True
    # 记录初始内存，用于计算净增量
    start_memory_rss = process.memory_info().rss
    print(f"✅ psutil导入成功，已开始监控内存。初始RSS: {start_memory_rss / 1024 ** 2:.2f} MB")
except ImportError:
    print("⚠️ 警告：无法导入 'psutil' 库。将无法统计内存使用情况。")
    print("   请通过 'pip install psutil' 命令安装。")
# =================================================================


# =================================================================
#  <<< 新增代码：解决 Matplotlib 中文乱码问题 >>>
# =================================================================
# 推荐使用 'SimHei' (黑体)，它在大多数 Windows 系统中都可用
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像时负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False
# =================================================================

# =================================================================
# <<< 新增代码：使用 try...finally 结构包裹主逻辑 >>>
# =================================================================
try:
    try:
        from SarOptMatch.evaluate_with_confidence import run_confidence_evaluation_with_entropy
    except ImportError:
        print(
            "❌ 错误：无法导入 run_confidence_evaluation_with_entropy。请确保 SarOptMatch.evaluate_with_confidence.py 文件存在且函数已正确定义。")


        def run_confidence_evaluation_with_entropy(*args, **kwargs):
            print("⚠️ 警告：run_confidence_evaluation_with_entropy 未成功导入，跳过基于熵值的评估。")
            return None, None

    try:
        from SarOptMatch.net_shift_visualizer import visualize_large_images_with_net_ransac_shift

        print("✅ Successfully imported visualize_large_images_with_net_ransac_shift from SarOptMatch.")
        # ... (保留您的调试打印)
    except ImportError as e_viz:
        print(f"⚠️ Warning: Could not import visualize_large_images_with_net_ransac_shift from SarOptMatch: {e_viz}")
        print("   Skipping large image net shift visualization.")


        def visualize_large_images_with_net_ransac_shift(*args, **kwargs):
            pass

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ {len(gpus)} GPU(s) found and memory growth set.")
        except RuntimeError as e:
            print(f"GPU设置错误: {e}")
    else:
        print("⚠️ 未找到GPU，将使用CPU。")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    TRAIN_MODEL = False
    PRETRAINED_WEIGHTS_DIR = "weights"
    MODEL_OPERATIONS_DIR = "lambda_models"
    ENTROPY_EVALUATION_OUTPUT_DIR = "evaluation_results_grouped_by_entropy"
    ORIGINAL_MASTER_DIR = r"E:\zhanghan\master_ims4"
    ORIGINAL_SLAVE_DIR = r"E:\zhanghan\slave_ims4"

    main_dataset_paths = [r'E:\OSdataset\OSdataset-5\OSdataset\256']
    main_ims_per_folder_list = [9623]

    EXTERNAL_VALIDATION_DATASET_PATH = r'E:\zhanghan\qietu_4\offset_pos32'
    EXTERNAL_VALIDATION_IMS_PER_FOLDER = 8370
    EXTERNAL_VALIDATION_BATCH_SIZE = 4

    success_datasets = []
    failed_datasets = []
    n_filters_list = [32]
    ENTROPY_PERCENTILE_FOR_SELECTION = 20
    LOG_HIGHEST_ENTROPY_PERCENTILE = 100 - ENTROPY_PERCENTILE_FOR_SELECTION

    # NEW: 定义统一的采样策略列表
    # 'conv': 编码器用卷积下采样，解码器用转置卷积上采样
    # 'interp_pool': 编码器用池化下采样，解码器用双线性插值上采样
    SAMPLING_STRATEGIES = ['conv']

    # 主数据集处理循环
    for dataset_path, ims_per_folder in tqdm(zip(main_dataset_paths, main_ims_per_folder_list),
                                             total=len(main_dataset_paths),
                                             desc="🛠️ 主数据集处理进度",
                                             file=sys.stdout,
                                             leave=True):
        dataset_name = os.path.basename(dataset_path)
        logger = None

        for n_filters in n_filters_list:
            # NEW: 循环遍历不同的统一采样策略
            for sampling_strategy_choice in SAMPLING_STRATEGIES:
                # 根据策略确定编码器和解码器的具体方法
                if sampling_strategy_choice == 'conv':
                    encoder_use_conv_down = True
                    decoder_upsampling_method = 'transpose_conv'
                    strategy_str_for_id = "conv_all"  # 用于文件名和日志
                elif sampling_strategy_choice == 'interp_pool':
                    encoder_use_conv_down = False  # 使用池化
                    decoder_upsampling_method = 'bilinear'
                    strategy_str_for_id = "interp_pool_all"
                else:
                    print(f"⚠️ 未知的采样策略: {sampling_strategy_choice}。跳过此策略。")
                    continue

                # MODIFIED: 更新 current_model_id 以反映统一的采样策略
                current_model_id = f"{dataset_name}_nfilt{n_filters}_strat_{strategy_str_for_id}"
                model_ready_for_evaluation = False

                if logger is not None and hasattr(logger, 'handlers'):
                    for handler in logger.handlers[:]:
                        handler.close()
                        logger.removeHandler(handler)
                logger = setup_logger(current_model_id)
                print(
                    f"\n🔍 正在处理主数据集: {dataset_name} | 每个文件夹图像数: {ims_per_folder} | n_filters = {n_filters} | 采样策略: {sampling_strategy_choice} (EncDownConv: {encoder_use_conv_down}, DecUp: {decoder_upsampling_method})")
                print(f"日志文件: logs/{current_model_id}.log")

                try:
                    print(f"--- 📂 加载和分割主数据集: {dataset_name} ---")
                    sar_files, opt_files, offsets = SarOptMatch.dataset.sen1_2(
                        data_path=dataset_path,
                        seed=seed,
                        ims_per_folder=ims_per_folder
                    )

                    TRAIN_BATCH_SIZE = 4
                    training_data, main_validation_data, main_validation_dataRGB = SarOptMatch.dataset.split_data(
                        sar_files, opt_files, offsets,
                        batch_size=TRAIN_BATCH_SIZE,
                        seed=seed,
                        masking_strategy="unet"
                    )
                    print(f"主数据集: {len(sar_files)} 个样本. 训练集和主验证集 (7:3) 已创建。")

                    matcher_init_config = {
                        'backbone': 'UnetPlus',
                        'n_filters': n_filters,
                        'activation': 'relu',
                        'use_conv_downsampling': encoder_use_conv_down,  # <--- NEW: 从策略中获取
                        'decoder_upsampling_method': decoder_upsampling_method,  # <--- NEW: 从策略中获取
                        'model_name': current_model_id
                        # 可以添加 star_mlp_ratio, star_drop_path_rate 等其他 UnetPlus 参数
                    }
                    matcher = SarOptMatch.architectures.SAR_opt_Matcher(**matcher_init_config)

                    matcher.create_model(
                        reference_im_shape=(256, 256, 1),
                        floating_im_shape=(192, 192, 1),
                        model_id_for_name=current_model_id
                    )

                    history_object = None
                    best_model_saver = None

                    if TRAIN_MODEL:
                        print(f"模式: 🟢 训练模式已启用 (TRAIN_MODEL = True)")
                        print(
                            f"使用 n_filters = {n_filters}, 采样策略 = {sampling_strategy_choice} 在主数据集上进行训练 (Batch Size: {TRAIN_BATCH_SIZE})...")
                        best_model_saver = BestModelSaver(main_validation_data, matcher, dataset_name=current_model_id)
                        print("🚀 开始在主数据集上训练...")
                        history_object, early_stopped = matcher.train(
                            training_data,
                            main_validation_data,
                            epochs=1,
                            callbacks=[best_model_saver]
                        )
                        print("🏁 主数据集训练完成.")
                        if os.path.exists(os.path.join(MODEL_OPERATIONS_DIR, f"{current_model_id}_best_model.h5")):
                            model_ready_for_evaluation = True
                            print(f"✅ 训练完成，最佳模型应已保存到 {MODEL_OPERATIONS_DIR} 并加载到 matcher.model。")
                        else:
                            print(
                                f"⚠️ 训练完成，但未在 {MODEL_OPERATIONS_DIR} 找到预期的最佳模型文件 {current_model_id}_best_model.h5。评估可能使用最后一次迭代的模型。")
                            model_ready_for_evaluation = True

                        history_save_dir = PRETRAINED_WEIGHTS_DIR
                        os.makedirs(history_save_dir, exist_ok=True)
                        history_save_path = os.path.join(history_save_dir, f"{current_model_id}_history.pkl")
                        try:
                            with open(history_save_path, 'wb') as f:
                                if history_object and hasattr(history_object, 'history'):
                                    pickle.dump(history_object.history, f)
                                    print(f"✅ 训练历史 (字典) 已保存到: {history_save_path}")
                        except Exception as e_hist_save:
                            print(f"⚠️ 保存训练历史失败: {e_hist_save}")

                        if history_object and best_model_saver:
                            score_history_to_plot = best_model_saver.score_history if hasattr(best_model_saver,
                                                                                              'score_history') else None
                            plot_training_curve(
                                model_name=current_model_id,
                                history_object=history_object,
                                score_history_list=score_history_to_plot
                            )

                    else:  # TRAIN_MODEL is False
                        print(f"模式: 🟡 加载模式已启用 (TRAIN_MODEL = False)")
                        best_model_filename = f"{current_model_id}_best_model.h5"
                        weights_path = os.path.join(MODEL_OPERATIONS_DIR, best_model_filename)
                        if os.path.exists(weights_path):
                            try:
                                print(f"⏳ 正在从 {weights_path} 加载预训练权重到 matcher.model...")
                                matcher.model.load_weights(weights_path)
                                print(f"✅ 成功加载预训练权重到 matcher.model。")
                                model_ready_for_evaluation = True
                            except Exception as e_load:
                                print(f"⚠️ 加载预训练权重到 matcher.model 失败: {e_load}")
                        else:
                            print(f"⚠️ 未找到预训练权重文件: {weights_path} (在 {MODEL_OPERATIONS_DIR} 中)")

                    if not model_ready_for_evaluation:
                        print(f"🔴 模型 ({current_model_id}) 未能成功准备 (训练或加载)。跳过此配置的所有评估步骤。")
                        failed_datasets.append(f"{current_model_id} (模型准备失败)")
                        continue

                    print(
                        f"\n--- 📈 开始在 主验证集 ({dataset_name}) 上进行评估 (evaluate_best_model将从内部加载模型) ---")
                    try:
                        evaluate_best_model(matcher, main_validation_data, main_validation_dataRGB, current_model_id,
                                            extract_features=False)

                        if matcher.model:
                            main_val_entropy_log_id = f"{current_model_id}_main_val"
                            print(
                                f"\n--- 🛡️ 调用基于熵值的评估 (主验证集: {dataset_name}, 使用 matcher.model, Log ID: {main_val_entropy_log_id}) ---")
                            run_confidence_evaluation_with_entropy(
                                model_to_evaluate=matcher.model,
                                validation_data_for_confidence=main_validation_data,
                                model_name_for_log=main_val_entropy_log_id,
                                entropy_percentile_for_selection=ENTROPY_PERCENTILE_FOR_SELECTION,
                                log_highest_entropy_samples_percentile=LOG_HIGHEST_ENTROPY_PERCENTILE,
                                output_dir=ENTROPY_EVALUATION_OUTPUT_DIR,
                                apply_ransac_on_low_entropy=True,
                                ransac_reproj_threshold=3.0,
                                min_ransac_samples=4,
                                run_ransac_only_on_all_samples_mode=False
                            )
                        else:
                            print(f"⚠️ matcher.model 未定义或无效，跳过在主验证集上的基于熵值的评估。")
                    except Exception as e_eval_main:
                        print(f"⚠️ 在主验证集 ({dataset_name}) 上的评估失败: {str(e_eval_main)}")
                        traceback.print_exc()

                    if EXTERNAL_VALIDATION_DATASET_PATH and os.path.exists(EXTERNAL_VALIDATION_DATASET_PATH):
                        external_dataset_name = os.path.basename(EXTERNAL_VALIDATION_DATASET_PATH)
                        external_model_id_for_loading = current_model_id

                        print(
                            f"\n\n--- ✨ [计时开始: {time.strftime('%Y-%m-%d %H:%M:%S')}] 开始处理外部验证数据集: {external_dataset_name} (使用模型: {external_model_id_for_loading}) ---")
                        start_time_external_eval = time.time()
                        try:
                            print(f"--- 📂 加载外部验证数据集: {external_dataset_name} ---")
                            ext_sar_files, ext_opt_files, ext_offsets = SarOptMatch.dataset.sen1_2(
                                data_path=EXTERNAL_VALIDATION_DATASET_PATH,
                                seed=seed,
                                ims_per_folder=EXTERNAL_VALIDATION_IMS_PER_FOLDER
                            )
                            print(f"外部验证数据集: {len(ext_sar_files)} 个样本将用于评估。")

                            if ext_sar_files.size == 0:
                                print(f"⚠️ 外部验证数据集 {external_dataset_name} 为空，跳过评估。")
                            else:
                                external_val_data = SarOptMatch.dataset.files_to_dataset(
                                    ext_opt_files, ext_sar_files, ext_offsets,
                                    masking_strategy="unet",
                                    batch_size=EXTERNAL_VALIDATION_BATCH_SIZE,
                                    convert_to_grayscale=True
                                )
                                print(
                                    f"\n--- 📈 开始在 外部验证集 ({external_dataset_name}) 上进行评估 (evaluate_best_model将从内部加载模型) ---")
                                evaluate_best_model(matcher, external_val_data, None,
                                                    external_model_id_for_loading,
                                                    extract_features=False)

                                if matcher.model:
                                    # --- 关键修复：为全局RANSAC和熵值评估创建包含所有样本的单个批次数据集 ---
                                    print(
                                        f"   准备外部验证集全局评估数据：创建包含全部 {len(ext_sar_files)} 个样本的单个批次...")
                                    external_val_data_for_global_eval = SarOptMatch.dataset.files_to_dataset(
                                        ext_opt_files, ext_sar_files, ext_offsets,
                                        masking_strategy="unet",
                                        batch_size=EXTERNAL_VALIDATION_BATCH_SIZE,  # 将所有样本放入一个批次
                                        convert_to_grayscale=True,

                                    )
                                    print("   外部验证集单批次数据准备完成。")
                                    # --- 修复结束 ---
                                    ext_val_entropy_log_id = f"{current_model_id}_on_ext_{external_dataset_name}"
                                    print(
                                        f"\n--- 🛡️ 调用基于熵值的评估 (外部验证集: {external_dataset_name}, 使用 matcher.model, Log ID: {ext_val_entropy_log_id}) ---")
                                    run_confidence_evaluation_with_entropy(
                                        model_to_evaluate=matcher.model,
                                        validation_data_for_confidence=external_val_data_for_global_eval,
                                        # <--- 使用新创建的数据集
                                        model_name_for_log=ext_val_entropy_log_id,
                                        entropy_percentile_for_selection=ENTROPY_PERCENTILE_FOR_SELECTION,
                                        log_highest_entropy_samples_percentile=LOG_HIGHEST_ENTROPY_PERCENTILE,
                                        output_dir=ENTROPY_EVALUATION_OUTPUT_DIR,
                                        apply_ransac_on_low_entropy=True,
                                        ransac_reproj_threshold=3.0,
                                        min_ransac_samples=4,
                                        run_ransac_only_on_all_samples_mode=False
                                    )

                                    base_id_for_ransac = current_model_id.replace(dataset_name, "ds")
                                    ext_name_short = external_dataset_name[:4]
                                    # MODIFIED: RANSAC 日志 ID 现在也反映统一的采样策略
                                    ext_val_ransac_only_log_id = f"{base_id_for_ransac}_ext_{ext_name_short}_{strategy_str_for_id}_RANSAC"

                                    print(
                                        f"\n--- 🛡️✨ 调用仅RANSAC评估 (外部验证集: {external_dataset_name}, 使用 matcher.model, Log ID: {ext_val_ransac_only_log_id}) ---")
                                    start_time_ransac_only_eval = time.time()
                                    # --- 关键修复：为全局RANSAC创建包含所有样本的单个批次数据集 ---
                                    print(
                                        f"   准备全局RANSAC数据：创建包含全部 {len(ext_sar_files)} 个样本的单个批次...")
                                    external_val_data_for_ransac_only = SarOptMatch.dataset.files_to_dataset(
                                        ext_opt_files, ext_sar_files, ext_offsets,
                                        masking_strategy="unet",
                                        batch_size=len(ext_sar_files),  # 将所有样本放入一个批次
                                        convert_to_grayscale=True,

                                    )
                                    # --- 修复结束 ---

                                    collected_samples_ransac_only, M_affine_global_from_ransac = run_confidence_evaluation_with_entropy(
                                        model_to_evaluate=matcher.model,
                                        validation_data_for_confidence=external_val_data_for_global_eval,
                                        # <--- 复用这个数据集
                                        model_name_for_log=ext_val_ransac_only_log_id,
                                        entropy_percentile_for_selection=0,
                                        log_highest_entropy_samples_percentile=0,
                                        apply_ransac_on_low_entropy=False,
                                        ransac_reproj_threshold=3.0,
                                        min_ransac_samples=4,
                                        run_ransac_only_on_all_samples_mode=True,
                                        output_dir=ENTROPY_EVALUATION_OUTPUT_DIR
                                    )
                                    end_time_ransac_only_eval = time.time()
                                    duration_ransac_only_eval = end_time_ransac_only_eval - start_time_ransac_only_eval
                                    print(
                                        f"--- ⏱️✨ 仅RANSAC评估 ({ext_val_ransac_only_log_id}) 耗时: {duration_ransac_only_eval:.2f} 秒 ({duration_ransac_only_eval / 60:.2f} 分钟) ---")

                                    if M_affine_global_from_ransac is not None:
                                        print(
                                            f"\n--- 💠 准备使用Net RANSAC Shift可视化原始大图 (基于模型 {current_model_id}) ---")
                                        try:
                                            original_master_tifs = sorted(
                                                glob.glob(os.path.join(ORIGINAL_MASTER_DIR, "optical_*.tif")))
                                            original_slave_tifs = sorted(
                                                glob.glob(os.path.join(ORIGINAL_SLAVE_DIR, "sar_*.tif")))

                                            if not original_master_tifs or not original_slave_tifs:
                                                print("⚠️ 无法在指定的原始大图目录中找到 .tif 文件。跳过大图可视化。")
                                            else:
                                                idx_of_original_pair = 0
                                                if idx_of_original_pair < len(original_master_tifs) and \
                                                        idx_of_original_pair < len(original_slave_tifs):
                                                    original_large_opt_path = original_master_tifs[idx_of_original_pair]
                                                    original_large_sar_path = original_slave_tifs[idx_of_original_pair]
                                                    if os.path.exists(original_large_opt_path) and os.path.exists(
                                                            original_large_sar_path):
                                                        artificial_offset_xy_cropping = (32, 32)
                                                        output_subdir_for_viz = os.path.join(
                                                            ENTROPY_EVALUATION_OUTPUT_DIR,
                                                            ext_val_ransac_only_log_id)
                                                        os.makedirs(output_subdir_for_viz, exist_ok=True)
                                                        # MODIFIED: 可视化文件名也反映统一的采样策略
                                                        large_img_viz_filename = f"large_img_net_shift_viz_strat_{strategy_str_for_id}.png"
                                                        large_img_viz_full_path = os.path.join(output_subdir_for_viz,
                                                                                               large_img_viz_filename)
                                                        visualize_large_images_with_net_ransac_shift(
                                                            original_optical_large_img_path=original_large_opt_path,
                                                            original_sar_large_img_path=original_large_sar_path,
                                                            M_affine_global_ransac=M_affine_global_from_ransac,
                                                            artificial_offset_xy=artificial_offset_xy_cropping,
                                                            output_visualization_path=large_img_viz_full_path,
                                                            checkerboard_block_size=128
                                                        )
                                                    else:
                                                        print(f"⚠️ 选定的原始大图路径不存在。跳过大图可视化。")
                                                else:
                                                    print(
                                                        f"⚠️ 索引 {idx_of_original_pair} 超出原始大图文件列表范围。跳过大图可视化。")
                                        except Exception as e_large_viz:
                                            print(f"⚠️ 调用大图可视化时发生错误: {e_large_viz}")
                                            traceback.print_exc()
                                    else:
                                        print("⚠️ 未能从RANSAC评估中获取有效的全局仿射矩阵。跳过大图可视化。")
                                else:
                                    print(f"⚠️ matcher.model 未定义或无效，跳过在外部验证集上的基于熵值的评估。")
                        except Exception as e_eval_ext:
                            print(f"⚠️ 在外部验证集 ({external_dataset_name}) 上的评估失败: {str(e_eval_ext)}")
                            traceback.print_exc()
                        finally:
                            end_time_external_eval = time.time()
                            duration_external_eval = end_time_external_eval - start_time_external_eval
                            print(
                                f"--- ✨ [计时结束: {time.strftime('%Y-%m-%d %H:%M:%S')}] 外部验证数据集 ({external_dataset_name}) 处理完成。")
                            print(
                                f"--- ⏱️  外部验证数据集 ({external_dataset_name}) 总耗时 (包括所有评估类型): {duration_external_eval:.2f} 秒 ({duration_external_eval / 60:.2f} 分钟) ---")
                    elif EXTERNAL_VALIDATION_DATASET_PATH:
                        print(f"⚠️ 未找到外部验证数据集路径: {EXTERNAL_VALIDATION_DATASET_PATH}，跳过外部验证。")

                    success_datasets.append(current_model_id)
                except Exception as e_main_loop:
                    print(
                        f"❌ 出错啦！主数据集 {dataset_name} (n_filters={n_filters}, strategy={sampling_strategy_choice}) 处理出现错误：{str(e_main_loop)}")
                    traceback.print_exc()
                    failed_datasets.append(f"{current_model_id} (主循环异常)")
                finally:
                    if logger is not None and hasattr(logger, 'handlers'):
                        for handler in logger.handlers[:]:
                            handler.close()
                            logger.removeHandler(handler)
                    K.clear_session()
                    gc.collect()
                    print(f"--- ✅ 主数据集处理完成: {current_model_id} ---")

        if logger is not None and hasattr(logger, 'handlers') and logger.handlers:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            # MODIFIED: 确保在最外层循环的最后关闭 logger
            print(
                f"关闭了主数据集 {dataset_name} (n_filters={n_filters_list[-1]}, strategy={SAMPLING_STRATEGIES[-1]}) 的logger。")

    print("\n\n=== 🏁 处理总结 🏁 ===")
    print(f"✅ 成功处理的配置 (主流程): {success_datasets}")
    print(f"❌ 失败的配置 (主流程): {failed_datasets}")

# =================================================================
# <<< 新增代码：在脚本结束时报告总体时间和内存使用 >>>
# =================================================================
finally:
    script_end_time = time.time()
    total_duration_seconds = script_end_time - script_start_time

    print("\n\n" + "=" * 20 + " 性能统计报告 " + "=" * 20)
    print(f"📊 脚本总运行时间: {total_duration_seconds:.2f} 秒 ({total_duration_seconds / 60:.2f} 分钟)")

    if psutil_available:
        # 获取脚本结束时的内存使用情况
        end_memory_rss = process.memory_info().rss
        net_memory_increase = end_memory_rss - start_memory_rss

        print("\n--- 内存使用情况说明 ---")
        print("内存使用是通过 'psutil' 库计算的。我们主要关注 RSS (Resident Set Size)。")
        print("RSS 指的是进程在物理内存(RAM)中占用的部分，是衡量实际内存消耗的常用指标。")
        print("由于Python的动态内存管理，精确的“峰值”内存难以在不侵入代码的情况下捕获。")
        print("我们在此报告脚本运行结束时的最终内存占用，它能很好地反映大规模数据处理后的内存状态。")
        print("--------------------------")
        print(f"🧠 初始内存占用 (RSS): {start_memory_rss / 1024 ** 2:.2f} MB")
        print(f"🧠 最终内存占用 (RSS): {end_memory_rss / 1024 ** 2:.2f} MB")
        print(f"🧠 净增内存占用 (RSS): {net_memory_increase / 1024 ** 2:.2f} MB")

        # 将字节转换为更易读的单位
        if end_memory_rss > 1024 ** 3:
            print(f"   (最终内存占用约为: {end_memory_rss / 1024 ** 3:.2f} GB)")
        else:
            print(f"   (最终内存占用约为: {end_memory_rss / 1024 ** 2:.2f} MB)")
    else:
        print("\n🧠 内存使用情况：未统计 (需要安装 'psutil' 库)。")
    print("=" * 55)

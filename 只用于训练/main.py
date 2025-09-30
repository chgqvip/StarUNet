import os
import tensorflow as tf
import numpy as np
import random
import traceback
import gc
import sys
from tqdm import tqdm
import SarOptMatch  # 主包
from SarOptMatch.train_modules import BestModelSaver, evaluate_best_model, plot_training_curve
from SarOptMatch.log import setup_logger  # 导入正确的 setup_logger
from tensorflow.keras import backend as K
from SarOptMatch.loss_module import make_loss, LossComponentsLogger
import SarOptMatch.dataset as dataset_module
import subprocess
import tracemalloc
# <<< MODIFIED END >>>
import glob
# ... (其余代码)

# === ✅ 是否训练模型，False 则只加载权重做推理 ===
train = False

script_dir = os.path.dirname(os.path.abspath(__file__))

# === 网格搜索参数定义 (使用 alpha_prime) ===
ALPHA_PRIME_LIST = [0,0.25]
THRESHOLD_LIST = [2]
# ALPHA_PRIME_LIST = [0,0.25,0.5,0.75,1]
# THRESHOLD_LIST = [2,4,6]
# === GPU 设置 ===
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === 固定随机种子 ===
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# === 数据集路径 ===
# dataset_paths = ['E:\sen1-2dataset',r'E:\OSdataset\OSdataset-5\OSdataset\256','E:\AOI_11_Rotterdam\TEST',]
# ims_per_folder_list = [100,9623,11164]
dataset_paths = ['E:\sen1-2dataset',]
ims_per_folder_list = [100]
# dataset_paths = [r'E:\OSdataset\OSdataset-5\OSdataset\256']
# ims_per_folder_list = [9623]
# dataset_paths = ['E:\AOI_11_Rotterdam\TEST',]
# ims_per_folder_list = [11164]
# Monkey patch dataset 中的 blur
def monkey_patch_threshold(thresh):
    original_blur = dataset_module.gaussian_blur

    def patched_blur(threshold_tensor, offsets):
        if not isinstance(thresh, tf.Tensor):
            threshold_to_use = tf.constant(float(thresh), dtype=tf.float32)
        else:
            threshold_to_use = tf.cast(thresh, dtype=tf.float32)
        return original_blur(threshold_to_use, offsets)

    dataset_module.gaussian_blur = patched_blur


# === 清空旧的 grid_results.csv 文件 ===
csv_path = os.path.join(script_dir, "grid_results.csv")  # script_dir 而不是 os.path.dirname(__file__)
# 在 main.py 的顶部，CSV 初始化部分：
if train:
    print(f"🧹 初始化/清空 grid_results.csv 文件...")
    with open(csv_path, "w") as f:
        pass # 只清空文件，不写入任何内容


# === 校准函数 (获取 S_ce 和 S_mse) ===
def calibrate_scaling_factors(dataset_path_for_cal, ims_per_folder_for_cal, seed_val, base_config, cal_threshold_val):
    print("\n--- 🚀 开始校准 S_ce 和 S_mse ---")
    s_ce_val = 3.0
    s_mse_val = 0.5

    monkey_patch_threshold(cal_threshold_val)
    print(f"校准时使用的 Threshold: {cal_threshold_val}")

    try:
        cal_ims_count = max(20, ims_per_folder_for_cal // 10)
        print(f"校准使用图像数: {cal_ims_count} from {dataset_path_for_cal}")
        sar_files_cal, opt_files_cal, offsets_cal = SarOptMatch.dataset.sen1_2(
            data_path=dataset_path_for_cal, seed=seed_val, ims_per_folder=cal_ims_count
        )
        # ⬇️ **修改点 1: 移除 split_data 调用中的 val_split 参数**
        _, val_data_cal, _ = SarOptMatch.dataset.split_data(  # _ 表示不使用 training_data_cal
            sar_files_cal, opt_files_cal, offsets_cal,
            batch_size=4, seed=seed_val, masking_strategy="unet"  # 移除了 val_split
        )
        if val_data_cal is None or tf.data.experimental.cardinality(val_data_cal).numpy() == 0:
            print("❌ 校准时验证数据为空，无法继续校准。")
            return s_ce_val, s_mse_val
    except Exception as e:
        print(f"❌ 校准时数据加载失败: {e}")
        traceback.print_exc()
        return s_ce_val, s_mse_val

    # --- 校准 S_ce ---
    print("\n--- 校准 S_ce (alpha_prime = 0.0) ---")
    try:
        K.clear_session()
        gc.collect()
        matcher_cal_ce = SarOptMatch.architectures.SAR_opt_Matcher()
        matcher_cal_ce.create_model(**base_config)
        matcher_cal_ce.model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                                     loss=make_loss(alpha_prime=0.0, S_ce=1.0, S_mse=1.0))
        temp_logger_ce = LossComponentsLogger(validation_data=val_data_cal, S_ce=1.0, S_mse=1.0)
        matcher_cal_ce.model.fit(val_data_cal.take(10), epochs=3, verbose=0, callbacks=[temp_logger_ce])
        if temp_logger_ce.ce_raw_history:
            s_ce_val = np.mean(temp_logger_ce.ce_raw_history[-2:])
            print(f"✅ 校准得到 S_ce 估计值: {s_ce_val:.4f}")
        else:
            print("⚠️ 校准 S_ce 时未获取到 ce_raw_history。")
        del matcher_cal_ce, temp_logger_ce
    except Exception as e:
        print(f"❌ 校准 S_ce 失败: {e}")
        traceback.print_exc()

    # --- 校准 S_mse ---
    print("\n--- 校准 S_mse (alpha_prime = 1.0) ---")
    try:
        K.clear_session()
        gc.collect()
        matcher_cal_mse = SarOptMatch.architectures.SAR_opt_Matcher()
        matcher_cal_mse.create_model(**base_config)
        matcher_cal_mse.model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                                      loss=make_loss(alpha_prime=1.0, S_ce=1.0, S_mse=1.0))
        temp_logger_mse = LossComponentsLogger(validation_data=val_data_cal, S_ce=1.0, S_mse=1.0)
        matcher_cal_mse.model.fit(val_data_cal.take(10), epochs=3, verbose=0, callbacks=[temp_logger_mse])
        if temp_logger_mse.mse_sum_raw_history:
            s_mse_val = np.mean(temp_logger_mse.mse_sum_raw_history[-2:])
            print(f"✅ 校准得到 S_mse 估计值: {s_mse_val:.4f}")
        else:
            print("⚠️ 校准 S_mse 时未获取到 mse_sum_raw_history。")
        del matcher_cal_mse, temp_logger_mse
    except Exception as e:
        print(f"❌ 校准 S_mse 失败: {e}")
        traceback.print_exc()

    final_s_ce = max(float(s_ce_val), 1e-5)
    final_s_mse = max(float(s_mse_val), 1e-5)
    print(f"--- ✅ 校准完成: S_ce = {final_s_ce:.4f}, S_mse = {final_s_mse:.4f} ---")
    K.clear_session()
    gc.collect()
    return final_s_ce, final_s_mse


# === 在主循环开始前执行校准 ===
S_CE_CALIBRATED = None
S_MSE_CALIBRATED = None
MODEL_CONFIG = {
    'reference_im_shape': (256, 256, 1), 'floating_im_shape': (192, 192, 1),
    'normalize': True, 'backbone': 'UnetPlus', 'n_filters': 32,
    'star_mlp_ratio': 4, 'star_drop_path_rate': 0.0,
    'use_conv_downsampling': True, 'decoder_upsampling_method': 'transpose_conv',
    'multiscale': False, 'attention': False, 'activation': 'relu'
}
CALIBRATION_THRESHOLD = THRESHOLD_LIST[0] if THRESHOLD_LIST else 2.0

if train and dataset_paths:
    print(f"使用数据集 '{dataset_paths[0]}' 和 threshold={CALIBRATION_THRESHOLD} 进行校准...")
    S_CE_CALIBRATED, S_MSE_CALIBRATED = calibrate_scaling_factors(
        dataset_paths[0], ims_per_folder_list[0], seed, MODEL_CONFIG, CALIBRATION_THRESHOLD
    )
    if S_CE_CALIBRATED is None or S_MSE_CALIBRATED is None or S_CE_CALIBRATED < 1e-5 or S_MSE_CALIBRATED < 1e-5:
        print("❌ 校准失败或值过小，将使用默认 S_ce=1.0, S_mse=1.0。结果可能不理想。")
        S_CE_CALIBRATED = 1.0
        S_MSE_CALIBRATED = 1.0

# === 主循环 ===
for dataset_path, ims_per_folder in tqdm(zip(dataset_paths, ims_per_folder_list),
                                         total=len(dataset_paths),
                                         desc="🛠️ 数据集处理进度",
                                         file=sys.__stdout__,  # tqdm 输出到标准输出
                                         leave=False):
    success_datasets = []
    failed_datasets = []
    base_name = os.path.basename(dataset_path)

    for alpha_prime in ALPHA_PRIME_LIST:
        for threshold in THRESHOLD_LIST:
            current_run_description = f"alpha_prime={alpha_prime:.2f}, threshold={threshold}"
            print(f"\n🔬 当前参数组合: {current_run_description}")  # 这个print会被logger捕获

            monkey_patch_threshold(threshold)

            ap_str = f"{alpha_prime:.2f}".replace('.', 'p')
            param_suffix = f"ap{ap_str}_t{threshold}"
            dataset_name_for_log = f"{base_name}_{param_suffix}"  # 用于日志和模型名称

            # ⬇️ **修改点 2: setup_logger 调用**
            # StreamLogger 会接管 sys.stdout, 所以 setup_logger 应该在其他 print 之前被调用，
            # 或者确保它的 print 语句是我们期望的日志初始化信息。
            # setup_logger 的 logs_dir 默认为 "logs"。如果想用 "SarOptMatch_logs"，则指定。
            # 这里我们使用 setup_logger 的默认 logs_dir="logs"
            logger_instance = setup_logger(dataset_name_for_log, logs_dir=os.path.join(script_dir, "SarOptMatch_logs"))
            # logger_instance 现在是 StreamLogger 的实例

            # 后续的 print 会通过 StreamLogger.write 输出到控制台和文件
            logger_instance.write(
                f"INFO: 开始处理: {dataset_name_for_log} | 每个文件夹图像数: {ims_per_folder}\n")  # 使用 logger.write
            logger_instance.write(f"INFO: 参数: alpha_prime={alpha_prime}, threshold={threshold}\n")
            if train:
                logger_instance.write(
                    f"INFO: 校准得到的缩放因子: S_ce={S_CE_CALIBRATED:.4f}, S_mse={S_MSE_CALIBRATED:.4f}\n")

            try:
                K.clear_session()
                gc.collect()

                logger_instance.write("INFO: 加载数据...\n")
                sar_files, opt_files, offsets = SarOptMatch.dataset.sen1_2(
                    data_path=dataset_path, seed=seed, ims_per_folder=ims_per_folder
                )
                training_data, validation_data, validation_dataRGB = SarOptMatch.dataset.split_data(
                    sar_files, opt_files, offsets, batch_size=4, seed=seed, masking_strategy="unet"
                )
                logger_instance.write("INFO: 数据加载完成。\n")

                matcher = SarOptMatch.architectures.SAR_opt_Matcher()
                matcher.create_model(**MODEL_CONFIG)
                matcher.model_name = dataset_name_for_log  # 使用与日志一致的名称
                logger_instance.write(f"INFO: 模型 {matcher.model_name} 创建完成。\n")

                if train:
                    initial_learning_rate = 0.0005
                    optimizer_for_training = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

                    current_s_ce_to_use = S_CE_CALIBRATED if S_CE_CALIBRATED is not None else 1.0
                    current_s_mse_to_use = S_MSE_CALIBRATED if S_MSE_CALIBRATED is not None else 1.0

                    loss_function_instance = make_loss(
                        alpha_prime=float(alpha_prime),
                        S_ce=float(current_s_ce_to_use),
                        S_mse=float(current_s_mse_to_use)
                    )
                    matcher.model.compile(optimizer=optimizer_for_training, loss=loss_function_instance)
                    logger_instance.write(
                        f"INFO: 模型编译完成，使用 alpha_prime={alpha_prime}, S_ce={current_s_ce_to_use:.4f}, S_mse={current_s_mse_to_use:.4f}\n")

                    loss_logger_callback = LossComponentsLogger(
                        validation_data=validation_data,
                        S_ce=float(current_s_ce_to_use),
                        S_mse=float(current_s_mse_to_use),
                        log_prefix='val_'
                    )
                    logger_instance.write("INFO: LossComponentsLogger 实例化完成。\n")

                    # 将 logger_instance (StreamLogger) 传递给 BestModelSaver (如果它支持)
                    # 假设 BestModelSaver 内部会使用 print()，这些 print 会被 StreamLogger 捕获
                    best_model_saver = BestModelSaver(validation_data, matcher,
                                                      dataset_name_for_log)  # 移除 logger_instance
                    all_callbacks = [best_model_saver, loss_logger_callback]

                    logger_instance.write("INFO: 开始训练...\n")
                    # Keras 的 fit 方法的 verbose 输出 (进度条等) 会被 StreamLogger 过滤
                    history, early_stopped = matcher.train(
                        training_data, validation_data, epochs=10, callbacks=all_callbacks
                    )
                    logger_instance.write(f"INFO: 训练完成。Early stopped: {early_stopped}\n")

                    # plot_training_curve 内部的 print 也会被捕获
                    plot_training_curve(dataset_name_for_log)  # 移除 logger_instance

                    with open(csv_path, "a") as f:
                        f.write(
                            f"{dataset_name_for_log},{alpha_prime:.2f},{threshold},{best_model_saver.best_score:.4f}\n")
                    logger_instance.write(f"INFO: 结果已写入 {csv_path}\n")

                    logger_instance.write("INFO: 加载最佳模型并评估...\n")
                    evaluate_best_model(matcher, validation_data, validation_dataRGB
                                       )  # 移除 logger_instance
                    success_datasets.append(dataset_name_for_log)
                    logger_instance.write("INFO: 评估完成。\n")

                else:  # 推理模式
                    weight_name = f"{dataset_name_for_log}_best_model.h5"
                    weight_path = os.path.join("lambda_models", weight_name)
                    logger_instance.write(f"INFO: 推理模式，尝试加载权重: {weight_path}\n")
                    if not os.path.isfile(weight_path):
                        logger_instance.write(f"ERROR: 未找到预训练权重文件: {weight_path}\n")  # 使用 logger.write
                        failed_datasets.append(dataset_name_for_log)
                    else:
                        matcher.model.load_weights(weight_path)
                        logger_instance.write(f"INFO: 成功加载预训练模型: {weight_path}\n")
                        evaluate_best_model(matcher, validation_data, validation_dataRGB,
                                            dataset_name_for_log)
                        success_datasets.append(dataset_name_for_log)
                        logger_instance.write("INFO: 推理评估完成。\n")

            except Exception as e:
                # 使用 logger_instance.write 记录异常
                error_msg = traceback.format_exc()
                logger_instance.write(f"ERROR: 数据集 {dataset_name_for_log} 处理时发生错误：\n{error_msg}\n")
                failed_datasets.append(dataset_name_for_log)
            finally:
                if 'logger_instance' in locals() and hasattr(logger_instance, 'close'):
                    logger_instance.write(f"INFO: 完成处理: {dataset_name_for_log}\n\n")
                    logger_instance.close()  # 关闭当前 StreamLogger，恢复 stdout
                K.clear_session()
                gc.collect()

    # === 总结输出 ===
    # 为总结创建一个新的、临时的 StreamLogger 或使用标准 print
    # 如果在这里再次调用 setup_logger，它会再次接管 stdout
    # 为了简单，这里的总结直接用 print，它会输出到控制台，但不会被特定的文件日志记录（除非有全局的 stdout 捕获）
    # 或者，您可以创建一个专门的总结日志文件
    summary_log_path = os.path.join(script_dir, "SarOptMatch_logs", f"{base_name}_summary.log")
    with open(summary_log_path, "a", encoding="utf-8") as summary_f:
        summary_f.write(f"\n=== 数据集 {base_name} 处理总结 ===\n")
        summary_f.write(f"✅ 成功处理的参数组合: {success_datasets}\n")
        summary_f.write(f"❌ 失败的参数组合: {failed_datasets}\n")
    print(f"\n=== 数据集 {base_name} 处理总结 ===")  # 也会输出到控制台
    print(f"✅ 成功处理的参数组合: {success_datasets}")
    print(f"❌ 失败的参数组合: {failed_datasets}")

    # === 调用 plot_grid_results.py 绘图 ===
    plot_script_path = os.path.join(script_dir, "SarOptMatch", "plot_grid_results.py")
    if not os.path.isfile(plot_script_path):
        print(f"❌ 找不到绘图脚本 plot_grid_results.py，期望路径: {plot_script_path}")
    else:
        try:
            print(f"📊 数据集 {base_name} 所有参数组合训练完成，生成热力图...")
            subprocess.run(["python", plot_script_path, "--csv_file", csv_path], check=True)
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")

print("\n🎉 所有数据集处理完毕。")

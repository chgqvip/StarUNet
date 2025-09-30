import tensorflow as tf


@tf.function
def crossEntropyNegativeMining(y_true, y_pred):
    """
    Computes the combined loss of the prediction.
    The negative mining term use a soft label.
    The cross entropy use a hard label with only 1 correct matching location.
    bs: batch_size
    """
    bs = tf.shape(y_pred)[0]
    
    # Predicted values inside correct matching region
    matching_region_samples = tf.divide(tf.reduce_sum(tf.multiply(y_pred,y_true),axis=(1,2)),tf.math.count_nonzero(y_true,axis=(1,2),keepdims=False,dtype=tf.float32))

    # Define non-matching region (only look outside correct matching region)
    negative_region = tf.multiply(y_pred,1-y_true)
    negative_region_filterd = tf.reshape(tf.where(tf.equal(negative_region,0.),1.,negative_region),[bs, 65*65])

    # Take n hardest negative samples from non-matching region
    n_neg_samples = 16
    neg_samples =  tf.reduce_mean(-tf.nn.top_k(-negative_region_filterd,k=n_neg_samples)[0],axis=-1) + 1

    # Negative Mining Term
    nm = tf.maximum(-(matching_region_samples-neg_samples),tf.constant(0,dtype=tf.float32))

    # Cross Entropy Term
    xent = tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(tf.where(tf.equal(y_true,tf.reduce_max(y_true,axis=(1,2),keepdims=True)),1.,0.),[bs,65*65]),
                                                   logits = tf.reshape(y_pred,[bs,65*65]))

    return xent + nm


@tf.function
def crossEntropy(y_true, y_pred):
    bs = tf.shape(y_pred)[0]
    # Matching can be regarded as a 2-D one-hot classification of 65x65 categories.
    return tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(tf.convert_to_tensor(y_true),[bs,65*65]),
                                                   logits = tf.reshape(y_pred,[bs,65*65]))


# --- 1. 修改后的损失函数工厂 ---
def make_loss(alpha_prime, S_ce, S_mse):
    """
    工厂函数，用于创建一个结合了归一化交叉熵 (Normalized CE)
    和归一化均方误差和 (Normalized MSE_sum) 的损失函数。

    L_norm_per_sample = (1 - alpha_prime) * (CE_per_sample / S_ce) + alpha_prime * (MSE_sum_per_sample / S_mse)

    Args:
        alpha_prime (float): 归一化 MSE_sum 项的权重因子, 理想范围 [0, 1]。
        S_ce (float): CE 的缩放因子 (典型值)。
        S_mse (float): MSE_sum 的缩放因子 (典型值)。

    Returns:
        一个与 Keras model.compile() 兼容的损失函数。
    """
    if not (S_ce > 1e-6 and S_mse > 1e-6):
        raise ValueError("S_ce and S_mse must be positive and non-negligible.")

    @tf.function
    def crossEntropyAddmeanSquaredError(y_true, y_pred):
        bs = tf.shape(y_pred)[0]
        y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred_logits = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        labels_flat = tf.reshape(y_true_tensor, [bs, -1])
        logits_flat = tf.reshape(y_pred_logits, [bs, -1])

        # --- CE 计算 (原始值) ---
        ce_per_sample_raw = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_flat,
            logits=logits_flat
        )  # 形状: [bs]

        # --- MSE_sum 计算 (原始值) ---
        y_pred_prob_flat = tf.nn.softmax(logits_flat, axis=-1)
        squared_error = tf.square(labels_flat - y_pred_prob_flat)
        # mse_sum_per_sample_raw 是每个样本的像素误差平方和
        mse_sum_per_sample_raw = tf.reduce_sum(squared_error, axis=1)  # 形状 [bs]

        # --- 归一化 ---
        ce_per_sample_norm = ce_per_sample_raw / S_ce
        mse_sum_per_sample_norm = mse_sum_per_sample_raw / S_mse

        # --- 组合归一化损失 ---
        # L_norm_per_sample = (1 - alpha_prime) * CE_norm + alpha_prime * MSE_norm
        combined_loss_per_sample = (1.0 - alpha_prime) * ce_per_sample_norm + \
                                   alpha_prime * mse_sum_per_sample_norm

        return combined_loss_per_sample

    return crossEntropyAddmeanSquaredError


def crossEntropyNegativeMiningAddmeanSquaredError(y_true, y_pred):
    """
    Computes the combined loss of the prediction.
    The negative mining term use a soft label.
    The cross entropy use a hard label with only 1 correct matching location.
    bs: batch_size
    """
    bs = tf.shape(y_pred)[0]

    # Predicted values inside correct matching region
    matching_region_samples = tf.divide(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=(1, 2)),
                                        tf.math.count_nonzero(y_true, axis=(1, 2), keepdims=False, dtype=tf.float32))

    # Define non-matching region (only look outside correct matching region)
    negative_region = tf.multiply(y_pred, 1 - y_true)
    negative_region_filterd = tf.reshape(tf.where(tf.equal(negative_region, 0.), 1., negative_region), [bs, 65 * 65])

    # Take n hardest negative samples from non-matching region
    n_neg_samples = 16
    neg_samples = tf.reduce_mean(-tf.nn.top_k(-negative_region_filterd, k=n_neg_samples)[0], axis=-1) + 1

    # Negative Mining Term
    nm = tf.maximum(-(matching_region_samples - neg_samples), tf.constant(0, dtype=tf.float32))

    # Cross Entropy Term
    xent = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(tf.where(tf.equal(y_true, tf.reduce_max(y_true, axis=(1, 2), keepdims=True)), 1., 0.),
                          [bs, 65 * 65]),
        logits=tf.reshape(y_pred, [bs, 65 * 65]))
    y_true_reshaped = tf.reshape(tf.convert_to_tensor(y_true), [bs, 65 * 65])
    y_pred_reshaped = tf.reshape(y_pred, [bs, 65 * 65])

    # 计算均方误差
    mse = tf.reduce_mean(tf.square(y_true_reshaped - y_pred_reshaped))

    return xent + nm + mse


@tf.function
def windowSum(optical_feature_map, SAR_template_size):
    """
    Returns the integral image of the optical feature map
    where SAR-template_size is the window size of the integral image 
    """
    
    x = tf.cumsum(optical_feature_map,axis=2)
    x = x[:,:,SAR_template_size:-1] -x[:,:,:-SAR_template_size-1]
    x = tf.cumsum(x,axis = 3)
    return (x[:,:,:,SAR_template_size:-1] - x[:,:,:,:-SAR_template_size-1])



@tf.function
def fft_layer(inputs):
    """
    FFT Cross Correlation
    """
    
    opt, sar = inputs
    fft_shape = tf.shape(opt)[2] + tf.shape(sar)[2] - 1
    fft_shape = [fft_shape, fft_shape]
    signal_element_mult = tf.multiply(tf.signal.rfft2d(opt,fft_shape), tf.signal.rfft2d(sar[:,:,::-1,::-1],fft_shape))
    return tf.signal.irfft2d(signal_element_mult,fft_shape)


@tf.function
def Normalization_layer(inputs):
    """
    Normalizes the similarity heatmap.
    Tensorflow implementation of the normalization process in scikit-image.match_template: 
    https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/feature/template.py#L31-L180

    """

    opt,sar,xcorr = inputs

    # Overflow thresholds
    ceil = tf.float32.max
    floor = tf.experimental.numpy.finfo(tf.float32).eps

    # SAR template shape
    float_image_shape = tf.shape(sar)

    # Zero-pad optical floating image to match the dimension of the FFT CC similarity map
    opt = tf.pad(opt,tf.constant([[0,0],[0,0],[192,192],[192,192]]),"CONSTANT")

    sar_volume = tf.cast(tf.math.reduce_prod(float_image_shape[2:]),dtype = tf.float32)
    sar_mean = tf.reduce_mean(sar,axis = [2,3])[:,:,tf.newaxis,tf.newaxis]
    sar_ssd = tf.math.reduce_sum((sar - sar_mean) ** 2, axis = [2,3])[:,:,tf.newaxis,tf.newaxis]

    # Compute integral images
    winsum = windowSum(opt,float_image_shape[2])
    winsum2 = windowSum(opt**2,float_image_shape[2])

    # Normalize
    numerator = tf.subtract(xcorr,winsum*sar_mean)
    winsum2 = tf.subtract(winsum2,tf.multiply(winsum,winsum)/sar_volume)
    winsum2 = tf.experimental.numpy.sqrt(tf.experimental.numpy.maximum(tf.multiply(winsum2,sar_ssd),0))

    # Clip values to avoid overflow
    mask = winsum2 > floor
    return tf.where(mask,tf.divide(tf.clip_by_value(numerator,floor,ceil) , tf.clip_by_value(winsum2,floor,ceil)) ,tf.zeros_like(xcorr))


import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback





# --- 2. 修改后的 LossComponentsLogger ---
class LossComponentsLogger(Callback):
    """
    在每个 epoch 结束时计算并记录原始及归一化的 CE 和 MSE_sum 损失分量。
    MSE_sum 是基于每个样本的像素误差平方和，然后对批次求平均。

    Args:
        validation_data (tf.data.Dataset): 用于计算损失分量的验证数据集。
        S_ce (float): CE 的缩放因子 (用于归一化显示)。
        S_mse (float): MSE_sum 的缩放因子 (用于归一化显示)。
        log_prefix (str): 添加到 TensorBoard/logs 中的指标名称前缀 (例如 'val_')。
    """

    def __init__(self, validation_data, S_ce, S_mse, log_prefix='val_'):
        super().__init__()
        self.validation_data = validation_data
        if not (S_ce > 1e-6 and S_mse > 1e-6):
            print(f"Warning: S_ce ({S_ce}) or S_mse ({S_mse}) is very small or zero. "
                  "Normalized components might be unstable or Inf/NaN.")
        self.S_ce = S_ce if S_ce > 1e-6 else 1.0  # Avoid division by zero in logging
        self.S_mse = S_mse if S_mse > 1e-6 else 1.0  # Avoid division by zero in logging
        self.log_prefix = log_prefix

        self.ce_raw_history = []
        self.mse_sum_raw_history = []
        self.ce_norm_history = []
        self.mse_norm_history = []

    def _calculate_components_and_normalized(self):
        all_y_true_np = []
        all_y_pred_np = []

        # print(f"\n[{self.__class__.__name__}] Calculating loss components on validation data...")
        for x_batch, y_batch_true_tensor in self.validation_data:
            y_batch_pred_np_direct = self.model.predict_on_batch(x_batch)
            # ... (rest of your data collection logic, same as before) ...
            if isinstance(y_batch_true_tensor, tf.Tensor):
                all_y_true_np.append(y_batch_true_tensor.numpy())
            elif isinstance(y_batch_true_tensor, np.ndarray):
                all_y_true_np.append(y_batch_true_tensor)
            else:
                try:
                    all_y_true_np.append(np.array(y_batch_true_tensor))
                except Exception as e:
                    print(f"Warning: Could not convert y_batch_true to NumPy array: {e}")
                    continue
            all_y_pred_np.append(y_batch_pred_np_direct)

        if not all_y_true_np or not all_y_pred_np:
            print(f"[{self.__class__.__name__}] Warning: Validation data seems empty. Cannot calculate components.")
            return None, None, None, None

        try:
            y_true_np_concat = np.concatenate(all_y_true_np, axis=0)
            y_pred_np_concat = np.concatenate(all_y_pred_np, axis=0)
        except ValueError as e:
            print(f"[{self.__class__.__name__}] Error concatenating arrays: {e}. Check batch shapes.")
            return None, None, None, None

        y_true_tf = tf.convert_to_tensor(y_true_np_concat, dtype=tf.float32)
        y_pred_tf = tf.convert_to_tensor(y_pred_np_concat, dtype=tf.float32)  # logits

        bs = tf.shape(y_pred_tf)[0]
        labels_flat = tf.reshape(y_true_tf, [bs, -1])
        logits_flat = tf.reshape(y_pred_tf, [bs, -1])

        # --- Raw CE Calculation ---
        ce_per_sample_raw = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_flat,
            logits=logits_flat
        )
        ce_raw_epoch_mean = tf.reduce_mean(ce_per_sample_raw)

        # --- Raw MSE_sum Calculation ---
        y_pred_prob_flat = tf.nn.softmax(logits_flat, axis=-1)
        squared_error = tf.square(labels_flat - y_pred_prob_flat)
        mse_sum_per_sample_raw = tf.reduce_sum(squared_error, axis=1)
        mse_sum_raw_epoch_mean = tf.reduce_mean(mse_sum_per_sample_raw)

        # --- Normalized Components Calculation ---
        ce_norm_epoch_mean = ce_raw_epoch_mean / self.S_ce
        mse_norm_epoch_mean = mse_sum_raw_epoch_mean / self.S_mse

        return (ce_raw_epoch_mean.numpy(), mse_sum_raw_epoch_mean.numpy(),
                ce_norm_epoch_mean.numpy(), mse_norm_epoch_mean.numpy())

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        results = self._calculate_components_and_normalized()
        if results[0] is not None:
            ce_raw, mse_sum_raw, ce_norm, mse_norm = results

            self.ce_raw_history.append(ce_raw)
            self.mse_sum_raw_history.append(mse_sum_raw)
            self.ce_norm_history.append(ce_norm)
            self.mse_norm_history.append(mse_norm)

            logs[self.log_prefix + 'ce_raw'] = ce_raw
            logs[self.log_prefix + 'mse_sum_raw'] = mse_sum_raw
            logs[self.log_prefix + 'ce_norm'] = ce_norm
            logs[self.log_prefix + 'mse_norm'] = mse_norm

            # Keras automatically logs the main loss as 'loss' and 'val_loss'
            # The 'val_loss' will be the result of normalized_combined_loss on validation data

            print(f"Epoch {epoch + 1}: "
                  f"{self.log_prefix}CE_raw: {ce_raw:.4f} | "
                  f"{self.log_prefix}MSE_sum_raw: {mse_sum_raw:.4f} | "
                  f"{self.log_prefix}CE_norm: {ce_norm:.4f} | "
                  f"{self.log_prefix}MSE_norm: {mse_norm:.4f}")
        else:
            print(f"Epoch {epoch + 1}: Could not calculate loss components for logging.")

    def on_train_end(self, logs=None):
        print("\n--- Loss Component Summary (Validation) ---")
        print("Epoch | CE (Raw) | MSE_sum (Raw) | CE (Norm) | MSE_sum (Norm)")
        print("-" * 65)
        for i in range(len(self.ce_raw_history)):
            print(f"{i + 1:<5} | {self.ce_raw_history[i]:<8.4f} | {self.mse_sum_raw_history[i]:<13.4f} | "
                  f"{self.ce_norm_history[i]:<9.4f} | {self.mse_norm_history[i]:<14.4f}")
        print("-" * 65)







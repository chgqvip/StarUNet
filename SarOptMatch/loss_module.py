import tensorflow as tf

# 辅助函数，用于从 y_true 中提取 mask_true
def _extract_mask_from_y_true(y_true_input):
    # 检查 y_true_input 是否是一个 Python 元组并且至少有一个元素
    # 注意：在 @tf.function 内部，直接用 isinstance(y_true_input, tuple) 可能不总是按预期工作，
    # 因为 y_true_input 可能是 tf.Tensor 占位符。
    # 但在 Keras 传递 y_true 时，如果它是一个元组，这里应该能检测到。
    # 一个更稳妥的方法是检查 y_true_input 是否是一个 TensorFlow 张量。
    # 如果它不是张量，并且可以被索引，那么它可能是我们期望的元组。
    # 然而，Keras 通常会将 Python 列表/元组的 y_true 结构传递给损失函数。

    # 尝试一个简单的方法：如果 y_true 是一个元组，并且其第一个元素是张量，则使用它。
    # 这种方法依赖于 Keras 如何传递 y_true。
    # 如果 y_true_input 是一个 Python 元组，并且第一个元素是 tf.Tensor
    if isinstance(y_true_input, tuple) and len(y_true_input) > 0 and isinstance(y_true_input[0], tf.Tensor):
        # print("DEBUG: y_true is a tuple, extracting first element as mask_true.")
        return y_true_input[0]
    # 如果 y_true_input 已经是张量，则直接使用
    elif isinstance(y_true_input, tf.Tensor):
        # print("DEBUG: y_true is already a tensor, using as mask_true.")
        return y_true_input
    else:
        # Fallback: 尝试将其转换为张量。如果 y_true_input 是一个元组，
        # 并且原始损失函数期望的是单个张量，那么原始的 tf.convert_to_tensor(y_true_input)
        # 在这里仍然会失败（如果元组不能直接转换）。
        # 这里的目标是，如果它是一个我们期望的 (mask, offsets) 元组，我们已经提取了 mask。
        # 如果它已经是 mask，我们也处理了。
        # 如果是其他意外情况，让原始的 convert_to_tensor 处理（并可能报错）。
        # print(f"DEBUG: y_true type is {type(y_true_input)}, attempting direct conversion or use.")
        return tf.convert_to_tensor(y_true_input) # 或者直接 return y_true_input，让后续的 convert_to_tensor 处理
@tf.function
def crossEntropyNegativeMining(y_true, y_pred):
    """
    Computes the combined loss of the prediction.
    The negative mining term use a soft label.
    The cross entropy use a hard label with only 1 correct matching location.
    bs: batch_size
    """
    # ✅ 提取 mask_true
    mask_true = _extract_mask_from_y_true(y_true)

    bs = tf.shape(y_pred)[0]

    # Predicted values inside correct matching region
    matching_region_samples = tf.divide(tf.reduce_sum(tf.multiply(y_pred, mask_true), axis=(1, 2)),
                                        tf.math.count_nonzero(mask_true, axis=(1, 2), keepdims=False, dtype=tf.float32))

    # Define non-matching region (only look outside correct matching region)
    negative_region = tf.multiply(y_pred, 1 - mask_true)
    negative_region_filterd = tf.reshape(tf.where(tf.equal(negative_region, 0.), 1., negative_region), [bs, 65 * 65])

    # Take n hardest negative samples from non-matching region
    n_neg_samples = 16
    neg_samples = tf.reduce_mean(-tf.nn.top_k(-negative_region_filterd, k=n_neg_samples)[0], axis=-1) + 1

    # Negative Mining Term
    nm = tf.maximum(-(matching_region_samples - neg_samples), tf.constant(0, dtype=tf.float32))

    # Cross Entropy Term
    xent = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(tf.where(tf.equal(mask_true, tf.reduce_max(mask_true, axis=(1, 2), keepdims=True)), 1., 0.),
                          [bs, 65 * 65]),
        logits=tf.reshape(y_pred, [bs, 65 * 65]))

    return xent + nm


@tf.function
def crossEntropy(y_true, y_pred):
    # ✅ 提取 mask_true
    mask_true = _extract_mask_from_y_true(y_true)

    bs = tf.shape(y_pred)[0]
    # Matching can be regarded as a 2-D one-hot classification of 65x65 categories.
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(tf.convert_to_tensor(mask_true), [bs, 65 * 65]), # 使用提取的 mask_true
        logits=tf.reshape(y_pred, [bs, 65 * 65]))


def crossEntropyAddmeanSquaredError(y_true, y_pred, alpha=0.1): # 移除了 @tf.function 以便 isinstance 工作
    # ✅ 提取 mask_true
    mask_true = _extract_mask_from_y_true(y_true)

    bs = tf.shape(y_pred)[0]
    ce = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(tf.convert_to_tensor(mask_true), [bs, 65 * 65]), # 使用提取的 mask_true
        logits=tf.reshape(y_pred, [bs, 65 * 65]))

    y_true_reshaped = tf.reshape(tf.convert_to_tensor(mask_true), [bs, 65 * 65]) # 使用提取的 mask_true
    y_pred_reshaped = tf.reshape(y_pred, [bs, 65 * 65])

    # 计算均方误差
    mse = tf.reduce_mean(tf.square(y_true_reshaped - y_pred_reshaped))
    return ce + alpha * mse


def crossEntropyNegativeMiningAddmeanSquaredError(y_true, y_pred): # 移除了 @tf.function
    # ✅ 提取 mask_true
    mask_true = _extract_mask_from_y_true(y_true)

    bs = tf.shape(y_pred)[0]

    # Predicted values inside correct matching region
    matching_region_samples = tf.divide(tf.reduce_sum(tf.multiply(y_pred, mask_true), axis=(1, 2)),
                                        tf.math.count_nonzero(mask_true, axis=(1, 2), keepdims=False, dtype=tf.float32))

    # Define non-matching region (only look outside correct matching region)
    negative_region = tf.multiply(y_pred, 1 - mask_true)
    negative_region_filterd = tf.reshape(tf.where(tf.equal(negative_region, 0.), 1., negative_region), [bs, 65 * 65])

    # Take n hardest negative samples from non-matching region
    n_neg_samples = 16
    neg_samples = tf.reduce_mean(-tf.nn.top_k(-negative_region_filterd, k=n_neg_samples)[0], axis=-1) + 1

    # Negative Mining Term
    nm = tf.maximum(-(matching_region_samples - neg_samples), tf.constant(0, dtype=tf.float32))

    # Cross Entropy Term
    xent = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(tf.where(tf.equal(mask_true, tf.reduce_max(mask_true, axis=(1, 2), keepdims=True)), 1., 0.),
                          [bs, 65 * 65]),
        logits=tf.reshape(y_pred, [bs, 65 * 65]))

    y_true_reshaped = tf.reshape(tf.convert_to_tensor(mask_true), [bs, 65 * 65]) # 使用提取的 mask_true
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


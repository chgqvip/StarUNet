import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout, concatenate, add, multiply, Input, \
    Conv2DTranspose, BatchNormalization, Lambda, Activation, UpSampling2D, Permute, Cropping2D
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Concatenate, Dropout, Add, Multiply, Input,
    Conv2DTranspose, BatchNormalization, Lambda, Activation, UpSampling2D, Permute,
    Cropping2D, Layer, ReLU, DepthwiseConv2D
)
from tensorflow.keras.layers import DepthwiseConv2D, Dense, GlobalAveragePooling2D, Reshape, Add, Activation, \
    LayerNormalization, Permute, Softmax
from tensorflow.keras import backend as K
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, Model
from . import loss_module
from . import utils
from tensorflow.keras import activations
from tensorflow.keras.callbacks import LambdaCallback
import SarOptMatch
from .train_modules import BestModelSaver
import traceback
import gc
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Concatenate, Dropout, Add, Multiply, Input,
    Conv2DTranspose, BatchNormalization, Lambda, Activation, UpSampling2D, Permute,
    Cropping2D, Layer, ReLU, DepthwiseConv2D
)
from tensorflow.keras.models import Model
# 假设 UnetPlus, UEncoder, StarEncoderBlock, StarDecoderBlock, ConvBNReLU, StarBlockKeras 已定义
# 并且 UnetPlus 和 StarDecoderBlock 支持 decoder_upsampling_method / upsampling_method
# 假设 loss_module 也已定义
try:
    from tensorflow_addons.layers import StochasticDepth
    print("INFO: Using tensorflow_addons.layers.StochasticDepth for DropPath.")
    HAS_TFA = True
except ImportError:
    print("WARNING: tensorflow_addons.layers.StochasticDepth not found. Using Dropout for DropPath if rate > 0.")
    HAS_TFA = False

def safe_predict(model, data, batch_size=32, verbose=0, steps=None, model_name="model"): # Added model_name for better logging
    """
    Safely predicts using the model, handling potential OOM errors for NumPy inputs
    by trying smaller batch sizes. For tf.data.Dataset, it relies on the Dataset's batching.
    """
    print(f"🧠 [{model_name}.safe_predict] Received data of type: {type(data)}")
    if isinstance(data, tf.data.Dataset):
        print(f"  Input is tf.data.Dataset. Element spec: {data.element_spec}")
        # For tf.data.Dataset, Keras model.predict should use the Dataset's own batching.
        # Explicitly setting batch_size might be ignored or lead to unexpected behavior if the dataset is already batched.
        # The 'steps' argument determines how many batches from the dataset to process.
        # If 'steps' is None, it processes the entire dataset.
        print(f"  Calling {model_name}.predict(data, steps={steps}, verbose={verbose})")
        try:
            preds = model.predict(data, steps=steps, verbose=verbose)
            print(f"  ✅ [{model_name}.safe_predict] Prediction successful for tf.data.Dataset.")
            return preds
        except tf.errors.ResourceExhaustedError as e:
            print(f"  ❌ [{model_name}.safe_predict] OOM Error during prediction with tf.data.Dataset: {e}")
            print(f"     Dataset element_spec: {data.element_spec}")
            print(f"     Consider reducing the batch size within the Dataset itself if OOM persists.")
            raise e # Re-raise OOM for Datasets as batch size is pre-defined
        except Exception as ex:
            print(f"  ❌ [{model_name}.safe_predict] Other error during prediction with tf.data.Dataset: {ex}")
            import traceback
            traceback.print_exc()
            raise ex

    elif isinstance(data, (np.ndarray, list)) or (isinstance(data, tuple) and all(isinstance(d, np.ndarray) for d in data)):
        # Handle single NumPy array or list/tuple of NumPy arrays (for multi-input models)
        num_samples = data[0].shape[0] if isinstance(data, (list, tuple)) else data.shape[0]
        print(f"  Input is NumPy array(s) with {num_samples} samples. Initial batch_size={batch_size}.")
        current_batch_size = batch_size
        while current_batch_size >= 1:
            try:
                print(f"  Trying {model_name}.predict with batch_size: {current_batch_size}")
                preds = model.predict(data, batch_size=current_batch_size, steps=steps, verbose=verbose)
                print(f"  ✅ [{model_name}.safe_predict] Prediction successful for NumPy array(s) with batch_size: {current_batch_size}.")
                return preds
            except tf.errors.ResourceExhaustedError as e:
                print(f"  ⚠️ [{model_name}.safe_predict] OOM Error with batch_size {current_batch_size}: {e}")
                if current_batch_size == 1:
                    print(f"  ❌ [{model_name}.safe_predict] OOM Error even with batch_size=1. Cannot proceed.")
                    raise e
                current_batch_size //= 2
                print(f"     Retrying with batch_size: {current_batch_size}")
            except Exception as ex:
                print(f"  ❌ [{model_name}.safe_predict] Other error during prediction with NumPy array(s): {ex}")
                import traceback
                traceback.print_exc()
                raise ex
        # Should not be reached if OOM at bs=1 is handled
        return None
    else:
        print(f"  Input type {type(data)} not explicitly handled for iterative batching. Using direct predict.")
        try:
            preds = model.predict(data, verbose=verbose)
            print(f"  ✅ [{model_name}.safe_predict] Prediction successful for unhandled data type.")
            return preds
        except Exception as ex:
            print(f"  ❌ [{model_name}.safe_predict] Error during prediction with unhandled data type: {ex}")
            import traceback
            traceback.print_exc()
            raise ex
# -----------------------------------------------------------------------------
# 辅助函数和基础模块 (ConvBNReLU, StarBlockKeras - 保持不变)
# -----------------------------------------------------------------------------
class ConvBNReLU(Layer):
    def __init__(self, c_out, kernel_size, stride=1, padding='same', activation=True, name=None, **kwargs):
        super(ConvBNReLU, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(
            filters=c_out,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            use_bias=False,
            name=f"{name}_conv" if name else None
        )
        self.bn = BatchNormalization(name=f"{name}_bn" if name else None)
        self.relu = ReLU(name=f"{name}_relu" if name else None)
        self.activation = activation
        self.c_out_arg = c_out
        self.kernel_size_arg = kernel_size
        self.stride_arg = stride
        self.padding_arg = padding
        self.activation_arg = activation

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.activation:
            x = self.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "c_out": self.c_out_arg,
            "kernel_size": self.kernel_size_arg,
            "stride": self.stride_arg,
            "padding": self.padding_arg,
            "activation": self.activation_arg,
        })
        return config

class StarBlockKeras(Layer):
    def __init__(self, dim, mlp_ratio=4, drop_path_rate=0.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.mlp_ch = int(mlp_ratio * dim)
        self.drop_path_rate = drop_path_rate
        self.dwconv1 = DepthwiseConv2D(kernel_size=7, strides=1, padding='same', use_bias=False, name=f"{name}_dw1" if name else None)
        self.bn_dw1 = BatchNormalization(name=f"{name}_bn_dw1" if name else None)
        self.fc1 = Conv2D(filters=self.mlp_ch, kernel_size=1, use_bias=True, name=f"{name}_fc1" if name else None)
        self.fc2 = Conv2D(filters=self.mlp_ch, kernel_size=1, use_bias=True, name=f"{name}_fc2" if name else None)
        self.act = ReLU(max_value=6.0, name=f"{name}_relu6" if name else None)
        self.multiply = Multiply(name=f"{name}_mul" if name else None)
        self.fc_g = Conv2D(filters=dim, kernel_size=1, use_bias=False, name=f"{name}_fc_g" if name else None)
        self.bn_g = BatchNormalization(name=f"{name}_bn_g" if name else None)
        self.dwconv2 = DepthwiseConv2D(kernel_size=7, strides=1, padding='same', use_bias=True, name=f"{name}_dw2" if name else None)
        if drop_path_rate > 0.:
            if HAS_TFA:
                self.drop_path = StochasticDepth(survival_probability=1.0 - drop_path_rate, name=f"{name}_dp" if name else None)
            else:
                self.drop_path = Dropout(drop_path_rate, noise_shape=(None, 1, 1, 1), name=f"{name}_dp_dropout" if name else None)
        else:
            self.drop_path = tf.identity
        self.add = Add(name=f"{name}_add" if name else None)

    def call(self, inputs, training=None):
        input_tensor = inputs
        x = self.dwconv1(inputs)
        x = self.bn_dw1(x, training=training)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x_act = self.act(x1)
        x = self.multiply([x_act, x2])
        x = self.fc_g(x)
        x = self.bn_g(x, training=training)
        x = self.dwconv2(x)
        if self.drop_path_rate > 0.:
            if HAS_TFA: x_dropped = self.drop_path(x, training=training)
            elif isinstance(self.drop_path, Dropout): x_dropped = self.drop_path(x, training=training)
            else: x_dropped = self.drop_path(x)
        else: x_dropped = self.drop_path(x)
        output = self.add([input_tensor, x_dropped])
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "mlp_ratio": self.mlp_ch / self.dim if self.dim > 0 else 0, "drop_path_rate": self.drop_path_rate})
        return config

# -----------------------------------------------------------------------------
# StarEncoderBlock (固定使用1个StarBlockKeras)
# -----------------------------------------------------------------------------
class StarEncoderBlock(Layer):
    def __init__(self, cin, cout, mlp_ratio=4, drop_path_rate=0.0, name=None, **kwargs): # 移除了 num_star_blocks
        super().__init__(name=name, **kwargs)
        self.cin_arg = cin
        self.cout_arg = cout
        self.mlp_ratio_arg = mlp_ratio
        self.drop_path_rate_arg = drop_path_rate
        self.num_star_blocks_arg = 1 # 固定为1

        if cin != cout:
            self.entry_conv = ConvBNReLU(cout, kernel_size=1, stride=1, padding='same', activation=True, name=f"{name}_entry_cbnr" if name else None)
        else:
            self.entry_conv = tf.identity

        # 固定使用一个StarBlockKeras
        self.star_block = StarBlockKeras(dim=cout, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate,
                                         name=f"{name}_sb_0" if name else None)

    def call(self, inputs, training=None):
        if self.cin_arg != self.cout_arg:
            x = self.entry_conv(inputs, training=training)
        else:
            x = self.entry_conv(inputs)
        x = self.star_block(x, training=training) # 直接调用单个star_block
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "cin": self.cin_arg,
            "cout": self.cout_arg,
            "mlp_ratio": self.mlp_ratio_arg,
            "drop_path_rate": self.drop_path_rate_arg,
            # num_star_blocks is fixed to 1, no need to store if not configurable
        })
        return config

# -----------------------------------------------------------------------------
# 编码器 UEncoder (固定使用StarEncoderBlock)
# -----------------------------------------------------------------------------
class UEncoder(tf.keras.Model):
    def __init__(self, n_filters, star_mlp_ratio=4, star_drop_path_rate=0.0,
                 use_conv_downsampling=True, name=None, **kwargs): # 移除了 use_star_blocks
        super(UEncoder, self).__init__(name=name, **kwargs)
        self.n_filters_arg = n_filters
        self.star_mlp_ratio_arg = star_mlp_ratio
        self.star_drop_path_rate_arg = star_drop_path_rate
        self.use_conv_downsampling_arg = use_conv_downsampling

        # Stage 1
        self.res1 = StarEncoderBlock(3, n_filters, mlp_ratio=star_mlp_ratio, drop_path_rate=star_drop_path_rate, name="enc_res1")
        if use_conv_downsampling:
            self.downsample1 = ConvBNReLU(n_filters, kernel_size=3, stride=2, name="enc_down1")
        else:
            self.downsample1 = MaxPooling2D(2, name="enc_pool1")

        # Stage 2
        self.res2 = StarEncoderBlock(n_filters, n_filters * 2, mlp_ratio=star_mlp_ratio, drop_path_rate=star_drop_path_rate, name="enc_res2")
        if use_conv_downsampling:
            self.downsample2 = ConvBNReLU(n_filters * 2, kernel_size=3, stride=2, name="enc_down2")
        else:
            self.downsample2 = MaxPooling2D(2, name="enc_pool2")

        # Stage 3
        self.res3 = StarEncoderBlock(n_filters * 2, n_filters * 4, mlp_ratio=star_mlp_ratio, drop_path_rate=star_drop_path_rate, name="enc_res3")
        if use_conv_downsampling:
            self.downsample3 = ConvBNReLU(n_filters * 4, kernel_size=3, stride=2, name="enc_down3")
        else:
            self.downsample3 = MaxPooling2D(2, name="enc_pool3")

        # Stage 4
        self.res4 = StarEncoderBlock(n_filters * 4, n_filters * 8, mlp_ratio=star_mlp_ratio, drop_path_rate=star_drop_path_rate, name="enc_res4")
        if use_conv_downsampling:
            self.downsample4 = ConvBNReLU(n_filters * 8, kernel_size=3, stride=2, name="enc_down4")
        else:
            self.downsample4 = MaxPooling2D(2, name="enc_pool4")

        # Bottleneck (Stage 5)
        self.res5_bottleneck = StarEncoderBlock(n_filters * 8, n_filters * 16, mlp_ratio=star_mlp_ratio, drop_path_rate=star_drop_path_rate, name="enc_res5_bottleneck")

    def call(self, x, training=None):
        features = []
        s1 = self.res1(x, training=training); features.append(s1)
        p1 = self.downsample1(s1, training=training) if isinstance(self.downsample1, ConvBNReLU) else self.downsample1(s1)
        s2 = self.res2(p1, training=training); features.append(s2)
        p2 = self.downsample2(s2, training=training) if isinstance(self.downsample2, ConvBNReLU) else self.downsample2(s2)
        s3 = self.res3(p2, training=training); features.append(s3)
        p3 = self.downsample3(s3, training=training) if isinstance(self.downsample3, ConvBNReLU) else self.downsample3(s3)
        s4 = self.res4(p3, training=training); features.append(s4)
        p4 = self.downsample4(s4, training=training) if isinstance(self.downsample4, ConvBNReLU) else self.downsample4(s4)
        s5_bottleneck = self.res5_bottleneck(p4, training=training); features.append(s5_bottleneck)
        return features

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters_arg,
            "star_mlp_ratio": self.star_mlp_ratio_arg,
            "star_drop_path_rate": self.star_drop_path_rate_arg,
            "use_conv_downsampling": self.use_conv_downsampling_arg,
        })
        return config

# -----------------------------------------------------------------------------
# 解码器模块 StarDecoderBlock (保持不变, 因为它已经是期望的结构)
# -----------------------------------------------------------------------------
class StarDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, skip_channels, up_channels, out_channels,
                 star_mlp_ratio=4, star_drop_path_rate=0.0,
                 upsampling_method='bilinear', # 新增参数：'bilinear' 或 'transpose_conv'
                 name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.skip_channels_arg = skip_channels
        self.up_channels_arg = up_channels
        self.out_channels_arg = out_channels
        self.star_mlp_ratio_arg = star_mlp_ratio
        self.star_drop_path_rate_arg = star_drop_path_rate
        self.upsampling_method_arg = upsampling_method # 存储参数

        if self.upsampling_method_arg == 'bilinear':
            self.up_layer = tf.keras.layers.UpSampling2D(
                size=(2, 2), interpolation='bilinear',
                name=f"{name}_bilinear_upsample" if name else None
            )
        elif self.upsampling_method_arg == 'transpose_conv':
            # 转置卷积的输出通道数通常设置为其输入的通道数(up_channels)，或者直接是out_channels
            # 这里我们让它输出 up_channels，保持与bilinear上采样后通道数一致，
            # 后续的 entry_conv 负责将融合后的通道调整到 out_channels
            self.up_layer = tf.keras.Sequential([
                Conv2DTranspose(
                    filters=up_channels, # 输出通道数与输入一致，仅做空间上采样
                    kernel_size=(2, 2),  # 或者 (4,4) 等，配合步长实现2x上采样
                    strides=(2, 2),
                    padding='same',      # 确保输出尺寸是输入的两倍
                    use_bias=False,      # BN会处理偏置
                    name=f"{name}_transpose_conv" if name else None
                ),
                BatchNormalization(name=f"{name}_transpose_bn" if name else None),
                ReLU(name=f"{name}_transpose_relu" if name else None)
            ], name=f"{name}_transpose_upsample_block" if name else None)
        else:
            raise ValueError(f"Unsupported upsampling_method: {self.upsampling_method_arg}. "
                             f"Choose 'bilinear' or 'transpose_conv'.")

        # entry_conv 和 star_block 的定义保持不变
        # fused_channels = skip_channels + up_channels (理论上的输入通道数)
        # self.entry_conv 的输入通道数是 skip_channels + (up_layer输出的通道数)
        # 由于 up_layer (无论是bilinear还是transpose_conv) 的输出通道数都是 up_channels,
        # 所以 fused_channels 的计算仍然是 skip_channels + up_channels
        self.entry_conv = ConvBNReLU(out_channels, kernel_size=1, stride=1, padding='same', activation=True,
                                     name=f"{name}_entry_cbnr" if name else None)
        self.star_block = StarBlockKeras(dim=out_channels, mlp_ratio=star_mlp_ratio,
                                         drop_path_rate=star_drop_path_rate,
                                         name=f"{name}_sb" if name else None)

    def call(self, x_up_input, skip_input, training=None):
        # x_up_input 是来自上一级解码器或bottleneck的输出，通道数为 self.up_channels_arg
        if self.upsampling_method_arg == 'transpose_conv':
            # 转置卷积块包含BN，需要 training 参数
            x_up = self.up_layer(x_up_input, training=training)
        else: # bilinear upsampling
            x_up = self.up_layer(x_up_input) # UpSampling2D 不需要 training 参数

        target_height = tf.shape(skip_input)[1]
        target_width = tf.shape(skip_input)[2]
        x_up_resized = tf.image.resize_with_crop_or_pad(x_up, target_height, target_width)

        fusion = tf.concat([skip_input, x_up_resized], axis=-1,
                           name=f"{self.name}_fusion" if self.name else None)

        processed_fusion = self.entry_conv(fusion, training=training) # entry_conv 包含BN
        output = self.star_block(processed_fusion, training=training) # star_block 包含BN
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "skip_channels": self.skip_channels_arg,
            "up_channels": self.up_channels_arg,
            "out_channels": self.out_channels_arg,
            "star_mlp_ratio": self.star_mlp_ratio_arg,
            "star_drop_path_rate": self.star_drop_path_rate_arg,
            "upsampling_method": self.upsampling_method_arg, # 添加到config
        })
        return config


# -----------------------------------------------------------------------------
# 整体 UnetPlus 模型 (固定使用StarEncoder 和 StarDecoder)
# -----------------------------------------------------------------------------
# ... (假设 ConvBNReLU, StarBlockKeras, StarEncoderBlock, UEncoder 已经定义) ...
# ... (StarDecoderBlock 定义如上) ...

class UnetPlus(tf.keras.Model):
    def __init__(self, n_filters,
                 star_mlp_ratio=4, star_drop_path_rate=0.0,
                 use_conv_downsampling=True,
                 decoder_upsampling_method='bilinear', # 新增参数，用于解码器上采样
                 name=None, **kwargs):
        super(UnetPlus, self).__init__(name=name, **kwargs)
        self.n_filters_arg = n_filters
        self.star_mlp_ratio_arg = star_mlp_ratio
        self.star_drop_path_rate_arg = star_drop_path_rate
        self.use_conv_downsampling_arg = use_conv_downsampling
        self.decoder_upsampling_method_arg = decoder_upsampling_method # 存储参数

        self.u_encoder = UEncoder(n_filters,
                                  star_mlp_ratio=star_mlp_ratio,
                                  star_drop_path_rate=star_drop_path_rate,
                                  use_conv_downsampling=use_conv_downsampling,
                                  name="unet_encoder")

        up_ch_d1 = n_filters * 16; skip_ch_d1 = n_filters * 8; out_ch_d1 = n_filters * 8
        up_ch_d2 = out_ch_d1;     skip_ch_d2 = n_filters * 4; out_ch_d2 = n_filters * 4
        up_ch_d3 = out_ch_d2;     skip_ch_d3 = n_filters * 2; out_ch_d3 = n_filters * 2
        up_ch_d4 = out_ch_d3;     skip_ch_d4 = n_filters;     out_ch_d4 = n_filters

        # 在实例化 StarDecoderBlock 时传递 upsampling_method
        self.decoder1 = StarDecoderBlock(skip_ch_d1, up_ch_d1, out_ch_d1, star_mlp_ratio, star_drop_path_rate,
                                         upsampling_method=self.decoder_upsampling_method_arg, name="star_dec1")
        self.decoder2 = StarDecoderBlock(skip_ch_d2, up_ch_d2, out_ch_d2, star_mlp_ratio, star_drop_path_rate,
                                         upsampling_method=self.decoder_upsampling_method_arg, name="star_dec2")
        self.decoder3 = StarDecoderBlock(skip_ch_d3, up_ch_d3, out_ch_d3, star_mlp_ratio, star_drop_path_rate,
                                         upsampling_method=self.decoder_upsampling_method_arg, name="star_dec3")
        self.decoder4 = StarDecoderBlock(skip_ch_d4, up_ch_d4, out_ch_d4, star_mlp_ratio, star_drop_path_rate,
                                         upsampling_method=self.decoder_upsampling_method_arg, name="star_dec4")

    def call(self, inputs, training=None, return_features=False):
        if inputs.shape[-1] == 1 and inputs.shape[-1] != 3 :
             inputs = tf.repeat(inputs, 3, axis=-1)
        encoder_skips = self.u_encoder(inputs, training=training)
        if return_features:
            return encoder_skips
        s1, s2, s3, s4, s5_bottleneck = encoder_skips
        d1 = self.decoder1(s5_bottleneck, s4, training=training)
        d2 = self.decoder2(d1, s3, training=training)
        d3 = self.decoder3(d2, s2, training=training)
        d4 = self.decoder4(d3, s1, training=training)
        return d4

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters_arg,
            "star_mlp_ratio": self.star_mlp_ratio_arg,
            "star_drop_path_rate": self.star_drop_path_rate_arg,
            "use_conv_downsampling": self.use_conv_downsampling_arg,
            "decoder_upsampling_method": self.decoder_upsampling_method_arg, # 添加到config
        })
        return config




#


# %% Models



class SAR_opt_Matcher():
    def __init__(self, **config):
        print("<SAR_opt_Matcher> instantiated")
        # ATTRIBUTES
        self.backbone = config.get('backbone', 'UnetPlus')
        self.model = None
        self.model_name = config.get('model_name', "sar_opt_matcher_model")
        # UnetPlus 参数
        self.n_filters = config.get('n_filters', 32)
        self.star_mlp_ratio = config.get('star_mlp_ratio', 4)
        self.star_drop_path_rate = config.get('star_drop_path_rate', 0.0)
        self.use_conv_downsampling = config.get('use_conv_downsampling', True) # For UEncoder
        self.decoder_upsampling_method = config.get('decoder_upsampling_method', 'bilinear') # 新增: For UnetPlus -> StarDecoderBlock
        # SAR_opt_Matcher 特定参数
        self.activation = config.get('activation', 'relu') # For Conv2D after UnetPlus features
        self.multiscale = config.get('multiscale', False) # 似乎未使用
        self.attention = config.get('attention', False)   # 似乎未使用


    def set_attributes(self, config):
        # 这个方法可以用来在实例化后更新属性
        self.backbone = config.get('backbone', self.backbone)
        self.model_name = config.get('model_name', self.model_name)
        self.n_filters = config.get('n_filters', self.n_filters)
        self.star_mlp_ratio = config.get('star_mlp_ratio', self.star_mlp_ratio)
        self.star_drop_path_rate = config.get('star_drop_path_rate', self.star_drop_path_rate)
        self.use_conv_downsampling = config.get('use_conv_downsampling', self.use_conv_downsampling)
        self.decoder_upsampling_method = config.get('decoder_upsampling_method', self.decoder_upsampling_method) # 新增
        self.activation = config.get('activation', self.activation)
        self.multiscale = config.get('multiscale', self.multiscale)
        self.attention = config.get('attention', self.attention)


    def print_attributes(self):
        """ Print class attrbutes """
        print("\n---Printing class attributes:---")
        for attribute, value in self.__dict__.items():
            if hasattr(value, "shape"):
                print(f"{attribute} = {value.shape}")
            else:
                print(f"{attribute} = {value}")
        print("\n")

    def export_attributes(self):
        def to_dict(self_obj):
            return {
                "backbone": self_obj.backbone,
                "model_name": self_obj.model_name,
                "n_filters": self_obj.n_filters,
                "star_mlp_ratio": self_obj.star_mlp_ratio,
                "star_drop_path_rate": self_obj.star_drop_path_rate,
                "use_conv_downsampling": self_obj.use_conv_downsampling,
                "decoder_upsampling_method": self_obj.decoder_upsampling_method, # 新增
                "activation": self_obj.activation,
                "multiscale": self_obj.multiscale,
                "attention": self_obj.attention,
            }
        config_to_export = to_dict(self)
        # 考虑取消注释并实现JSON导出
        # import json
        # with open(self.model_name + '_config.json', 'w') as file:
        #     json.dump(config_to_export, file, indent=4)
        print(f"Attributes for {self.model_name} (config for export): {config_to_export}")


    def create_model(self, reference_im_shape=(256, 256, 1), floating_im_shape=(192, 192, 1),
                     normalize=True, # normalize 参数似乎没有在模型构建中使用
                     model_id_for_name=None, **config):

        # 优先使用传入的 config 更新属性，其次是 __init__ 中设置的
        current_config = self.__dict__.copy() # 获取当前所有属性
        current_config.update(config) # 用传入的 config 更新

        self.backbone = current_config.get('backbone')
        # UnetPlus 参数
        self.n_filters = current_config.get('n_filters')
        self.star_mlp_ratio = current_config.get('star_mlp_ratio')
        self.star_drop_path_rate = current_config.get('star_drop_path_rate')
        self.use_conv_downsampling = current_config.get('use_conv_downsampling')
        self.decoder_upsampling_method = current_config.get('decoder_upsampling_method') # 新增
        # SAR_opt_Matcher 特定参数
        self.activation = current_config.get('activation')

        if model_id_for_name:
            self.model_name = model_id_for_name
        elif 'model_name' in current_config and current_config['model_name'] is not None:
            self.model_name = current_config['model_name']


        opt_in = Input(shape = reference_im_shape, name="optical_input")
        sar_in = Input(shape = floating_im_shape, name="sar_input")

        float_im_size = floating_im_shape[0] - 1
        response_crop = ((float_im_size, float_im_size), (float_im_size, float_im_size))

        if self.backbone.lower() == "unetplus":
            unetplus_common_args = {
                "n_filters": self.n_filters,
                "star_mlp_ratio": self.star_mlp_ratio,
                "star_drop_path_rate": self.star_drop_path_rate,
                "use_conv_downsampling": self.use_conv_downsampling,
                "decoder_upsampling_method": self.decoder_upsampling_method, # 传递给 UnetPlus
            }

            ref_heatmap_generator = UnetPlus(**unetplus_common_args, name="ref_feature_extractor")
            float_heatmap_generator = UnetPlus(**unetplus_common_args, name="float_feature_extractor")

            # Process optical image
            opt_heatmap_features = ref_heatmap_generator(opt_in) # UnetPlus 直接返回 decoder4 输出
            opt_heatmap = Conv2D(8, 3, activation=self.activation, padding='same', kernel_initializer='HeNormal',
                                 name="psi_opt_o")(opt_heatmap_features)

            # Process SAR image
            sar_heatmap_features = float_heatmap_generator(sar_in) # UnetPlus 直接返回 decoder4 输出
            sar_heatmap = Conv2D(8, 3, activation=self.activation, padding='same', kernel_initializer='HeNormal',
                                 name="psi_SAR_o")(sar_heatmap_features)

            # 假设 loss_module 和 crossEntropyAddmeanSquaredError 已定义
            loss = loss_module.crossEntropyAddmeanSquaredError
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        # Realign channels for FFT
        sar_heatmap_permuted = Permute((3, 1, 2), name="sar_permute")(sar_heatmap)
        opt_heatmap_permuted = Permute((3, 1, 2), name="opt_permute")(opt_heatmap)

        # FFT-based Cross Correlation
        xcorr = Lambda(loss_module.fft_layer, name="fft_cross_correlation")([opt_heatmap_permuted, sar_heatmap_permuted])

        # Normalization
        xcorr_normalized = Lambda(loss_module.Normalization_layer, name="ncc_normalization")([opt_heatmap_permuted, sar_heatmap_permuted, xcorr])

        # Crop the Normalized Cross Correlation heatmap
        out_cropped = Cropping2D(cropping=response_crop, data_format="channels_first", name="ncc_crop")(xcorr_normalized)

        # Move channels back
        out_permuted_back = Permute((2, 3, 1), name="output_permute_back")(out_cropped)

        # Averagely reduce the channel number to 1 and sharpen
        out_final = Lambda(lambda x: tf.divide(tf.reduce_mean(x, axis=3, keepdims=True), (1 / 30)), name="output_sharpening")(out_permuted_back)

        model = Model(inputs=[opt_in, sar_in], outputs=out_final, name=self.model_name)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss=loss)

        self.model = model
        print(f"Model '{self.model_name}' created successfully.")
        model.summary(line_length=120)
        return model



    def export_model_architecture(self):
        """ Plot model architecture (pdf is also possible)"""
        if self.model is not None:
            tf.keras.utils.plot_model(self.model, to_file=self.model_name + '.png', show_shapes=True,
                                      show_layer_names=True)

    def train(self, training_data: tf.data.Dataset, validation_data: tf.data.Dataset, epochs=5, callbacks=[]):
        if utils.confirmCommand("Train a new model?"):
            print("--training")

            my_callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001,
                                                     verbose=1),
            ]
            my_callbacks.extend(callbacks)

            history = self.model.fit(training_data, epochs=epochs,
                                     validation_data=validation_data,
                                     callbacks=my_callbacks)

            early_stopped = False
            for cb in my_callbacks:
                if isinstance(cb, tf.keras.callbacks.EarlyStopping) and cb.stopped_epoch > 0:
                    early_stopped = True
                    print(f"⛔ Internal EarlyStopping triggered at epoch {cb.stopped_epoch + 1}")

            print("--Saving weights")
            os.makedirs("weights", exist_ok=True)
            self.model.save_weights(f"weights/{self.model_name}.h5")

            print("--Saving history")
            final_history = history.history.copy()

            # ✅ 添加 score 历史（若存在）
            for cb in my_callbacks:
                if isinstance(cb, BestModelSaver):
                    final_history['score'] = cb.score_history

            with open(f"weights/{self.model_name}_history", 'wb') as file:
                pickle.dump(final_history, file)

            self.export_attributes()
            return history, early_stopped

    def load_model(self):
        # It seems like the support for exotic kinds of Python functions within the Lambda(...)
        # doesn't play very nicely with Keras serialization.
        # So we only save the weights. Then we create the architecture from the code and load the weights.

        print("--Loading")
        self.model_name = utils.open_file("Model to load:")

        # Find the .json file
        with open(os.path.splitext(self.model_name)[0] + ".json", 'r') as f:
            config = json.load(f)

        # Set attributes
        self.set_attributes(config)

        # Create the architecture and load stored weights
        self.create_model(**config)
        self.model.load_weights(self.model_name)

    def plot_history(self, ):
        with open(utils.open_file(self.model_name), 'rb') as f:
            history = pickle.load(f)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])

    def predict_heatmap(self, validation_data: tf.data.Dataset):
        if self.model is not None:
            print("--Calculating heatmaps")
            ncc_heatmap = self.model.predict(validation_data)
            if hasattr(ncc_heatmap, "shape"):
                if len(ncc_heatmap.shape) == 4:
                    ncc_heatmap = np.squeeze(ncc_heatmap, axis=-1)

            return ncc_heatmap
        else:
            raise ("Model not defined")

    # SarOptMatch/architectures.py

    # SarOptMatch/architectures.py

    def calculate_features(self, validation_data: tf.data.Dataset,
                           batch_size_for_feature_extraction: int = 4):
        if self.model is None:
            print("❌ [calculate_features] Model not defined.")
            raise RuntimeError("Model not defined")

        print(f"ℹ️ [calculate_features] Starting feature map calculation.")
        print(
            f"  Note: batch_size_for_feature_extraction ({batch_size_for_feature_extraction}) parameter is currently NOT directly used for batching if validation_data is already batched.")

        print(f"DEBUG [calculate_features] Initial validation_data.element_spec: {validation_data.element_spec}")

        # =======================================================================================
        # === 步骤 1: 定义和填充 feature_map_outputs ===
        # =======================================================================================
        layer_names_in_model = [layer.name for layer in self.model.layers]
        feature_map_outputs = []
        required_primary_layers = ["psi_opt_o", "psi_SAR_o"]
        # 假设您的模型中没有 psi_opt_d, psi_SAR_d 这些层，或者您不打算提取它们
        # 如果有，请取消注释下一行或根据实际情况修改
        # multiscale_layers = ["psi_opt_d", "psi_SAR_d"]
        multiscale_layers = []  # 如果没有多尺度层，则为空列表

        missing_primary = False

        print(f"ℹ️ [calculate_features] Attempting to collect outputs for primary layers: {required_primary_layers}")
        for layer_name in required_primary_layers:
            found_layer_output = None
            try:
                layer_obj = self.model.get_layer(name=layer_name)
                found_layer_output = layer_obj.output
                print(f"  ✅ Found layer '{layer_name}' output via get_layer().")
            except ValueError:
                print(
                    f"  ⚠️ Layer '{layer_name}' not found via get_layer(). Trying direct name match in model.layers...")
                for layer in self.model.layers:
                    if layer.name == layer_name:
                        found_layer_output = layer.output
                        print(f"  ✅ Found layer '{layer_name}' output via direct name match.")
                        break

            if found_layer_output is not None:
                feature_map_outputs.append(found_layer_output)
            else:
                print(f"  ❌ Required primary layer '{layer_name}' not found in the model.")
                missing_primary = True

        if missing_primary:
            print(f"  Available layers in model '{self.model.name}': {layer_names_in_model}")
            return None

        print(
            f"ℹ️ [calculate_features] Primary features collected: {[l.name.split(':')[0].split('/')[0] for l in feature_map_outputs]}")

        if hasattr(self, 'multiscale') and self.multiscale and multiscale_layers:  # 仅当 multiscale_layers 非空时才尝试
            print("ℹ️ [calculate_features] Multiscale mode enabled. Attempting to extract multiscale features.")
            multiscale_outputs_to_add = []
            for layer_name in multiscale_layers:
                found_layer_output = None
                try:
                    layer_obj = self.model.get_layer(name=layer_name)
                    found_layer_output = layer_obj.output
                    print(f"  ✅ Found multiscale layer '{layer_name}' output via get_layer().")
                except ValueError:
                    print(
                        f"  ⚠️ Multiscale layer '{layer_name}' not found via get_layer(). Trying direct name match...")
                    for layer in self.model.layers:
                        if layer.name == layer_name:
                            found_layer_output = layer.output
                            print(f"  ✅ Found multiscale layer '{layer_name}' output via direct name match.")
                            break

                if found_layer_output is not None:
                    multiscale_outputs_to_add.append(found_layer_output)
                else:
                    print(f"  ⚠️ Optional multiscale layer '{layer_name}' not found.")

            if multiscale_outputs_to_add:
                feature_map_outputs.extend(multiscale_outputs_to_add)
                print(
                    f"ℹ️ [calculate_features] Added multiscale features: {[l.name.split(':')[0].split('/')[0] for l in multiscale_outputs_to_add]}")
            else:  # multiscale_outputs_to_add 为空
                print(
                    f"⚠️ [calculate_features] Multiscale mode was set, but no multiscale layers ({multiscale_layers}) were found or successfully added.")
        elif hasattr(self, 'multiscale') and self.multiscale and not multiscale_layers:
            print(
                "ℹ️ [calculate_features] Multiscale mode enabled, but 'multiscale_layers' list is empty. No multiscale features to extract.")
        else:
            print(
                "ℹ️ [calculate_features] Multiscale mode disabled or 'multiscale' attribute not found/False. Extracting only primary features.")

        if not feature_map_outputs:
            print("❌ [calculate_features] No valid feature map outputs could be collected (list is empty).")
            return None
        # =======================================================================================
        # === 步骤 1 结束 ===
        # =======================================================================================

        # =======================================================================================
        # === 步骤 2: 创建 visualization_model ===
        # =======================================================================================
        visualization_model = None
        try:
            print("ℹ️ [calculate_features] Creating visualization_model on CPU...")
            print(f"  DEBUG [calculate_features] Base model inputs for visualization_model: {self.model.inputs}")
            print(
                f"  DEBUG [calculate_features] Outputs for visualization_model (feature_map_outputs): {feature_map_outputs}")
            visualization_model = tf.keras.Model(inputs=self.model.inputs, outputs=feature_map_outputs)
            print("✅ [calculate_features] Visualization model created successfully on CPU.")
            print(f"  DEBUG [calculate_features] Visualization model inputs: {visualization_model.inputs}")
            print(f"  DEBUG [calculate_features] Visualization model outputs: {visualization_model.outputs}")
            print(f"  DEBUG [calculate_features] Visualization model summary:")
            visualization_model.summary(print_fn=lambda x: print(f"    {x}"))
        except Exception as e:
            print(f"❌ [calculate_features] Failed to create visualization_model: {e}")
            import traceback;
            traceback.print_exc();
            return None

        # =======================================================================================
        # === 步骤 3: 准备数据并预测 (应用字典映射) ===
        # =======================================================================================
        input_names_for_dict = []
        try:
            # 从 visualization_model 获取输入层的实际名称
            input_names_for_dict = visualization_model.input_names
            if len(input_names_for_dict) != 2:
                print(
                    f"❌ [calculate_features] Expected 2 input layer names from visualization_model.input_names, but got {len(input_names_for_dict)}: {input_names_for_dict}")
                print(
                    f"  Model inputs from visualization_model.inputs: {[inp.name for inp in visualization_model.inputs]}")
                if len(visualization_model.inputs) == 2:
                    input_names_for_dict = [visualization_model.inputs[0].name.split(':')[0],
                                            visualization_model.inputs[1].name.split(':')[0]]
                    print(f"  Using fallback names from .inputs: {input_names_for_dict}")
                else:
                    input_names_for_dict = ["input_1", "input_2"]  # 根据模型摘要，这应该是默认值
                    print(f"  Using hardcoded default names: {input_names_for_dict} (Please verify!)")

            # 根据模型摘要，我们期望 "input_1" 和 "input_2"
            # 确保获取到的名称与预期一致，或者至少打印警告
            expected_names = ["input_1", "input_2"]
            if input_names_for_dict != expected_names:
                print(
                    f"⚠️ [calculate_features] Input names from model are {input_names_for_dict}. Expected {expected_names} based on summary. Ensure data order (opt then SAR) matches these names.")

            print(
                f"ℹ️ [calculate_features] Using input layer names for dict mapping: '{input_names_for_dict[0]}' (for opt), '{input_names_for_dict[1]}' (for SAR)")

        except AttributeError:
            print(
                f"⚠️ [calculate_features] visualization_model.input_names attribute not found. Trying to get names from .inputs.")
            try:
                input_names_for_dict = [inp.name.split(':')[0] for inp in visualization_model.inputs]
                if len(input_names_for_dict) != 2:
                    print(
                        f"❌ [calculate_features] Fallback: Expected 2 input layer names, got {input_names_for_dict}. Using defaults.")
                    input_names_for_dict = ["input_1", "input_2"]
                print(
                    f"ℹ️ [calculate_features] Using input layer names from .inputs for dict mapping: {input_names_for_dict}")
            except Exception as e_input_names_fallback:
                print(
                    f"❌ [calculate_features] Error getting input layer names via fallback: {e_input_names_fallback}. Using defaults.")
                input_names_for_dict = ["input_1", "input_2"]

        if len(input_names_for_dict) != 2:
            print(
                f"CRITICAL ❌ [calculate_features] Could not reliably determine 2 input names for dict mapping. Aborting.")
            return None

        # 定义映射函数
        # 使用闭包来捕获 input_names_for_dict
        def get_map_to_dict_fn(names_list):
            def extract_model_inputs_as_dict(batched_inputs_tuple, batched_targets_tuple):
                return {
                    names_list[0]: batched_inputs_tuple[0],
                    names_list[1]: batched_inputs_tuple[1]
                }

            return extract_model_inputs_as_dict

        map_fn_to_dict = get_map_to_dict_fn(input_names_for_dict)
        input_dataset_for_predict = validation_data.map(map_fn_to_dict)

        print(
            f"DEBUG [calculate_features] After map to dict (input_dataset_for_predict.element_spec): {input_dataset_for_predict.element_spec}")

        try:
            for sample_element_dict in input_dataset_for_predict.take(1):
                print(f"  DEBUG [calculate_features] Sample from input_dataset_for_predict (dict):")
                if isinstance(sample_element_dict, dict):
                    for k, v in sample_element_dict.items():
                        print(f"    '{k}': shape {v.shape}, dtype {v.dtype}")
                else:
                    print(f"    Sample is not a dict: {type(sample_element_dict)}")
        except Exception as e_take_dict:
            print(f"  DEBUG [calculate_features] Error taking dict sample: {e_take_dict}")

        final_input_dataset_for_predict = input_dataset_for_predict.prefetch(
            tf.data.experimental.AUTOTUNE)

        print(
            f"ℹ️ [calculate_features] Final input dataset spec for safe_predict (dict mapped): {final_input_dataset_for_predict.element_spec}")
        print("ℹ️ [calculate_features] Calling safe_predict for feature extraction (with dict mapped dataset)...")
        # 在 calculate_features 中，调用 safe_predict 之前
        print("ℹ️ [calculate_features] Attempting prediction on CPU due to potential OOM on GPU...")
        try:
            with tf.device('/CPU:0'):  # 强制在 CPU 上执行
                extracted_features_list = safe_predict(  # safe_predict 内部的 model.predict 或 predict_on_batch 会在此上下文中运行
                    visualization_model,
                    final_input_dataset_for_predict,
                    verbose=1,  # 增加详细程度以观察进度
                    steps=None,
                    model_name="visualization_model_cpu"
                )
        except Exception as e_cpu_predict:
            print(f"❌ [calculate_features] Error during CPU prediction: {e_cpu_predict}")
            raise e_cpu_predict


        if extracted_features_list is None:
            print("❌ [calculate_features] safe_predict returned None. Feature extraction failed.")
            return None

        # Keras model.predict with multiple outputs returns a list of arrays.
        # If visualization_model has only one output, it might return a single array.
        # Ensure extracted_features_list is always a list of arrays.
        num_vis_outputs = len(visualization_model.outputs)
        if not isinstance(extracted_features_list, list) and num_vis_outputs == 1:
            extracted_features_list = [extracted_features_list]
        elif not isinstance(extracted_features_list, list) and num_vis_outputs > 1:
            # This shouldn't happen if safe_predict returns list for multi-output
            print(
                f"⚠️ [calculate_features] Expected a list of features from safe_predict for {num_vis_outputs} outputs, but got {type(extracted_features_list)}. Wrapping it.")
            extracted_features_list = [extracted_features_list]  # Tentative
        elif isinstance(extracted_features_list, list) and len(extracted_features_list) != num_vis_outputs:
            print(
                f"⚠️ [calculate_features] Mismatch: safe_predict returned {len(extracted_features_list)} feature sets, but visualization_model has {num_vis_outputs} outputs.")

        print(
            f"✅ [calculate_features] Feature maps extracted successfully. Number of feature sets: {len(extracted_features_list if extracted_features_list else [])}")
        return extracted_features_list





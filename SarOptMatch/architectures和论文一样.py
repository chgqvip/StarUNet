import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout, concatenate, add, multiply, Input, \
    Conv2DTranspose, BatchNormalization, Lambda, Activation, UpSampling2D, Permute, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Input, DepthwiseConv2D, MaxPooling2D, \
    Concatenate, UpSampling2D, Dropout, Lambda
from tensorflow.keras.layers import DepthwiseConv2D, Dense, GlobalAveragePooling2D, Reshape, Add, Activation, \
    LayerNormalization, Permute, Softmax
from tensorflow.keras import backend as K
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
class Conv2dReLU(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        super(Conv2dReLU, self).__init__()
        if padding == 0:
            padding_option = 'valid'
        else:
            padding_option = 'same'

        self.conv = Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding_option,
            use_bias=not use_batchnorm
        )
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNReLU(Layer):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding='same', activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = Conv2D(
            filters=c_out,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            use_bias=False
        )
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.activation = activation

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(Layer):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = tf.keras.Sequential([
            ConvBNReLU(cin, cout, 3, 1, padding='same', activation=True),
            ConvBNReLU(cout, cout, 3, 1, padding='same', activation=True)
        ])
        self.conv1 = Conv2D(cout, 1, use_bias=False, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.add = Add()

    def call(self, inputs):
        x = self.conv(inputs)
        h = x
        x = self.conv1(inputs)

        x = self.add([h, x])
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWCONV(Layer):
    """
    Depthwise Convolution
    """

    def __init__(self, in_channels, kernel_size=3, stride=1, padding='same', depth_multiplier=1):
        super(DWCONV, self).__init__()
        self.depthwise = DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            depth_multiplier=depth_multiplier,
            use_bias=True
        )

    def call(self, inputs):
        result = self.depthwise(inputs)
        return result


class UEncoder(tf.keras.Model):
    def __init__(self):
        super(UEncoder, self).__init__()
        self.res1 = DoubleConv(3, 32)
        self.pool1 = MaxPooling2D(2)
        self.res2 = DoubleConv(32, 64)
        self.pool2 = MaxPooling2D(2)
        self.res3 = DoubleConv(64, 128)
        self.pool3 = MaxPooling2D(2)
        self.res4 = DoubleConv(128, 256)
        self.pool4 = MaxPooling2D(2)
        self.res5 = DoubleConv(256, 512)
        # self.pool5 = MaxPooling2D(2)

    def call(self, x):
        features = []
        x = self.res1(x)
        features.append(x)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)
        x = self.pool3(x)

        x = self.res4(x)
        features.append(x)
        x = self.pool4(x)

        x = self.res5(x)
        features.append(x)
        # x = self.pool5(x)
        # features.append(x)
        return features



class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding='same', use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding='same', use_batchnorm=use_batchnorm)
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        # self.conv3 = Conv2D(out_channels, 1, use_bias=False, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = tf.concat([x, skip], axis=-1)  # 注意合并轴为 -1，表示最后的通道维
        # h=self.conv3(x)


        x = self.conv1(x)
        x = self.conv2(x)
        # x=h+x
        x = self.bn(x)
        x = self.relu(x)
        return x


class MLP(Layer):
    def __init__(self, dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.fc1 = Dense(dim * 4)
        self.fc2 = Dense(dim)
        self.act = activations.relu
        self.dropout = Dropout(0.1)

    @tf.function
    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x


class MultiScaleAtten(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(MultiScaleAtten, self).__init__()
        self.dim = dim
        self.num_head = 8
        self.scale = (dim // self.num_head) ** 0.5

        # 定义层
        self.qkv_linear = layers.Dense(dim * 3, use_bias=False)
        self.softmax = layers.Softmax(axis=-1)
        self.proj = layers.Dense(dim, use_bias=False)

    @tf.function
    def call(self, x):
        # 获取输入的维度信息
        shape = tf.shape(x)
        B, num_blocks, _, _, C = shape[0], shape[1], shape[2], shape[3], shape[4]

        # 生成 Q, K, V
        qkv = self.qkv_linear(x)
        qkv = tf.reshape(qkv, (B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head))
        print("Shape of qkv before transpose: ", qkv.shape)
        qkv = tf.transpose(qkv, perm=[4, 0, 1, 2, 5, 3, 6])
        print("Shape of qkv after transpose: ", qkv.shape)

        # 分离 Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        atten = tf.matmul(q, k, transpose_b=True)
        # 为了满足 Softmax 维度限制，合并额外的维度
        # 假设 atten 的形状是 (num_heads, B, num_blocks, num_blocks, _, _)
        # 我们合并 num_blocks 和 num_blocks
        original_shape = tf.shape(atten)
        # original_shape = atten.shape
        atten = tf.reshape(atten, (
            original_shape[0], original_shape[1], original_shape[2] * original_shape[3], original_shape[4],
            original_shape[5]))
        atten = self.softmax(atten)

        # 恢复原始维度
        atten = tf.reshape(atten, original_shape)

        # 应用注意力权重
        atten_value = tf.matmul(atten, v)
        atten_value = tf.transpose(atten_value, perm=[0, 1, 2, 4, 3, 5])

        # 重新整形回原始的尺寸
        atten_value = tf.reshape(atten_value, (B, num_blocks, num_blocks, -1, C))

        # 应用最终的投影
        atten_value = self.proj(atten_value)

        return atten_value





class InterTransBlock(Layer):
    def __init__(self, dim, **kwargs):
        super(InterTransBlock, self).__init__(**kwargs)
        self.SlayerNorm_1 = LayerNormalization(epsilon=1e-6)
        self.SlayerNorm_2 = LayerNormalization(epsilon=1e-6)
        self.Attention = MultiScaleAtten(dim)
        self.FFN = MLP(dim)
        self.add = Add()

    @tf.function
    def call(self, inputs):
        h = inputs  # (B, N, H)
        x = self.SlayerNorm_1(h)
        x = self.Attention(x)  # Apply attention
        x = self.add([h, x])  # Add residual connection

        h = x
        x = self.SlayerNorm_2(h)
        x = self.FFN(x)  # Apply feed-forward network
        x = self.add([h, x])  # Add residual connection

        return x


class SpatialAwareTrans(Model):
    def __init__(self, dim=128, num=1, depth=4, channels=[128, 256, 512, 1024]):
        super(SpatialAwareTrans, self).__init__()
        self.ini_win_size = 2
        self.channels = channels
        self.dim = dim
        self.depth = depth
        self.fc_module = [layers.Dense(self.dim) for _ in self.channels]
        self.fc_rever_module = [layers.Dense(channel) for channel in self.channels]
        self.group_attention = [InterTransBlock(dim) for _ in range(num)]
        self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]

    @tf.function
    def call(self, x):
        # Project channel dimension to dim
        x = [self.fc_module[i](item) for i, item in enumerate(x)]

        # Patch Matching
        for j, item in enumerate(x):
            shape = tf.shape(item)
            B, H, W, C = shape[0], shape[1], shape[2], shape[3]
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = tf.reshape(item, (B, H // win_size, win_size, W // win_size, win_size, C))
            item = tf.transpose(item, (0, 1, 3, 2, 4, 5))
            item = tf.reshape(item, (B, H // win_size, W // win_size, win_size * win_size, C))
            x[j] = item

        x = tf.concat(x, axis=-2)  # (B, H // win, W // win, N, C)

        # Scale fusion using group attention
        for attention_block in self.group_attention:
            x = attention_block(x)

        x = tf.split(x, self.split_list, axis=-2)

        # Patch reversion
        for j, item in enumerate(x):
            shape = tf.shape(item)
            B, num_blocks, _, N, C = shape[0], shape[1], shape[2], shape[3], shape[4]
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = tf.reshape(item, (B, num_blocks, num_blocks, win_size, win_size, C))
            item = tf.transpose(item, (0, 1, 3, 2, 4, 5))
            item = tf.reshape(item, (B, num_blocks * win_size, num_blocks * win_size, C))
            item = self.fc_rever_module[j](item)
            x[j] = item

        return x


class ParallEncoder(tf.keras.Model):
    def __init__(self):
        super(ParallEncoder, self).__init__()
        self.Encoder1 = UEncoder()  # 假定这是已经定义的一个类
        self.Encoder2 = TransEncoder()  # 假定这是已经定义的一个类
        self.fusion_module = []
        self.num_module = 3
        self.channel_list = [64, 128, 256]
        self.fusion_list = [128,256, 512]
        # self.inter_trans = SpatialAwareTrans(dim=128)  # 假定这是已经定义的一个类

        self.squeelayers = []
        for i in range(self.num_module):
            self.squeelayers.append(
                tf.keras.layers.Conv2D(self.fusion_list[i], (1, 1), padding='same')
            )

    def call(self, x):
        skips = []
        features = self.Encoder1(x)
        feature_trans = self.Encoder2(features)
        # feature_trans = self.inter_trans(feature_trans)

        skips.extend(features[:2])
        for i in range(self.num_module):
            skip = self.squeelayers[i](tf.concat([feature_trans[i], features[i + 2]], axis=-1))
            skips.append(skip)
        return skips






class TransEncoder(tf.keras.Model):
    def __init__(self):
        super(TransEncoder, self).__init__()
        self.block_layer = [2, 2, 2, 1]
        self.channels = [128,256, 512, 512]
        self.size = None  # 初始化时不设置尺寸


    def set_sizes(self, input_shape):
        # 根据输入尺寸动态选择尺寸列表
        if input_shape[1] == 256:
            self.size = [64, 32, 16, 8]
        elif input_shape[1] == 192:
            self.size = [48, 24, 12, 6]
        else:
            # 可以添加更多尺寸条件或抛出错误
            raise ValueError("Unsupported input size")
        self.build_stages()  # 根据设置的尺寸构建阶段

    def build_stages(self):
        if self.size is None:
            return  # 如果尺寸未设置，跳过构建

        # 重新构建所有阶段
        self.stage1 = tf.keras.Sequential([
            IntraTransBlock(img_size=self.size[0],
                            in_channels=self.channels[0],
                            stride=2,
                            d_h=self.channels[0] // 8,
                            d_v=self.channels[0] // 8,
                            d_w=self.channels[0] // 8,
                            num_heads=8)
            for _ in range(self.block_layer[0])
        ])

        self.stage2 = tf.keras.Sequential([
            IntraTransBlock(img_size=self.size[1],
                            in_channels=self.channels[1],
                            stride=2,
                            d_h=self.channels[1] //8,
                            d_v=self.channels[1] // 8,
                            d_w=self.channels[1] // 8,
                            num_heads=8)
            for _ in range(self.block_layer[1])
        ])

        self.stage3 = tf.keras.Sequential([
            IntraTransBlock(img_size=self.size[2],
                            in_channels=self.channels[2],
                            stride=2,
                            d_h=self.channels[2] //8,
                            d_v=self.channels[2] // 8,
                            d_w=self.channels[2] // 8,
                            num_heads=8)
            for _ in range(self.block_layer[2])
        ])

        self.stage4 = tf.keras.Sequential([
            IntraTransBlock(img_size=self.size[3],
                            in_channels=self.channels[3],
                            stride=1,
                            d_h=self.channels[3] // 8,
                            d_v=self.channels[3] // 8,
                            d_w=self.channels[3] // 8,
                            num_heads=8)
            for _ in range(self.block_layer[3])
        ])
        self.downlayers = [
            ConvBNReLU(self.channels[i], self.channels[i] * 2, kernel_size=2, stride=2, padding='valid')
            for i in range(len(self.block_layer) - 1)
        ]

        self.squeelayers = [
            tf.keras.layers.Conv2D(self.channels[i] * 2, kernel_size=1, strides=1, padding='same')
            for i in range(len(self.block_layer) - 2)
        ]

        self.squeeze_final = tf.keras.layers.Conv2D(self.channels[-1], kernel_size=1, strides=1, padding='same')

    def call(self, inputs):
        if self.size is None:
            # 如果尺寸未设置，从输入中自动设置尺寸
            self.set_sizes(inputs[0].shape)
        _, _, feature0, feature1, feature2, feature3 = inputs

        feature0_trans = self.stage1(feature0)
        feature0_trans_down = self.downlayers[0](feature0_trans)

        feature1_in = tf.concat([feature1, feature0_trans_down], axis=-1)
        feature1_in = self.squeelayers[0](feature1_in)
        feature1_trans = self.stage2(feature1_in)
        feature1_trans_down = self.downlayers[1](feature1_trans)

        feature2_in = tf.concat([feature2, feature1_trans_down], axis=-1)
        feature2_in = self.squeelayers[1](feature2_in)
        feature2_trans = self.stage3(feature2_in)
        feature2_trans_down = self.downlayers[2](feature2_trans)

        feature3_in = tf.concat([feature3, feature2_trans_down], axis=-1)
        feature3_in = self.squeeze_final(feature3_in)
        feature3_trans = self.stage4(feature3_in)

        return [feature0_trans, feature1_trans, feature2_trans, feature3_trans]



class ScaleFormer(tf.keras.Model):
    def __init__(self):
        super(ScaleFormer, self).__init__()
        self.u_encoder = UEncoder()

        self.decoder1 = DecoderBlock(768, 128)
        self.decoder2 = DecoderBlock(512, 64)
        self.decoder3 = DecoderBlock(256, 32)
        self.decoder_final = DecoderBlock(32, 32)

    def call(self, inputs, return_features=False):
        if inputs.shape[-1] == 1:
            inputs = tf.repeat(inputs, 3, axis=-1)
        encoder_skips = self.u_encoder(inputs)

        if return_features:
            return encoder_skips

        x1_up = self.decoder1(encoder_skips[-1], encoder_skips[-2])
        x2_up = self.decoder2(x1_up, encoder_skips[-3])
        x3_up = self.decoder3(x2_up, encoder_skips[-4])

        x_final = self.decoder_final(x3_up, encoder_skips[-5])
        return x_final  # 返回最终的特征图







# %% Models


class SAR_opt_Matcher():
    def __init__(self, **config):

        print("<SAR_opt_Matcher> instantiated")
        # ATTRIBUTES
        self.backbone = None


    def set_attributes(self, config):

        self.backbone = config.get('backbone')


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
        def to_dict(self):
            # Creates a dict of important attributes to export
            return {"backbone": self.backbone}

        config = to_dict(self)
        # with open(self.model_name + '.json', 'w') as file:
        #     json.dump(config, file)

    def create_model(self, reference_im_shape=(256, 256, 3), floating_im_shape=(192, 192, 1), normalize=True, **config):

        self.backbone = config.get('backbone')
        self.n_filters = config.get('n_filters')
        self.multiscale = config.get('multiscale')
        self.attention = config.get('attention')
        self.activation = config.get('activation')

        backbone = self.backbone
        n_filters = self.n_filters
        attention = self.attention
        multiscale = self.multiscale
        activation = self.activation

        # Assume shape given is in channel_last format, also assumes images are squares.
        # Define input shapes of input images
        # opt_in = Input(shape = reference_im_shape)
        # sar_in = Input(shape = floating_im_shape)
        batch_size =4
        reference_im_shape_with_batch = (batch_size,) + reference_im_shape
        floating_im_shape_with_batch = (batch_size,) + floating_im_shape

        # 使用显式批次大小
        opt_in = Input(batch_shape=reference_im_shape_with_batch)
        sar_in = Input(batch_shape=floating_im_shape_with_batch)
        # opt_in = tf.keras.layers.Lambda(lambda x: tf.ensure_shape(x, reference_im_shape_with_batch), name="reshape_opt")(opt_in)
        # sar_in = tf.keras.layers.Lambda(lambda x: tf.ensure_shape(x, floating_im_shape_with_batch), name="reshape_sar")(sar_in)

        float_im_size = floating_im_shape[0] - 1
        response_crop = ((float_im_size, float_im_size), (float_im_size, float_im_size))

        if backbone.lower() == "scaleformer":
            ref_heatmap_generator = ScaleFormer()

            float_heatmap_generator = ScaleFormer()

            # Process optical image
            opt_heatmap = ref_heatmap_generator(opt_in)
            opt_heatmap = Conv2D(8, 3, activation=self.activation, padding='same', kernel_initializer='HeNormal',
                                 name="psi_opt_o")(opt_heatmap)

            # Process SAR image
            sar_heatmap = float_heatmap_generator(sar_in)
            sar_heatmap = Conv2D(8, 3, activation=self.activation, padding='same', kernel_initializer='HeNormal',
                                 name="psi_SAR_o")(sar_heatmap)
            # loss = loss_module.crossEntropyNegativeMining
            # loss = loss_module.crossEntropy
            loss = loss_module.crossEntropyAddmeanSquaredError
            # loss =loss_module.crossEntropyNegativeMiningAddmeanSquaredError



        # Reallign channels for FFT
        sar_heatmap = Permute((3, 1, 2))(sar_heatmap)
        opt_heatmap = Permute((3, 1, 2))(opt_heatmap)

        # FFT-based Cross Correlation
        xcorr = Lambda(loss_module.fft_layer)([opt_heatmap, sar_heatmap])

        xcorr = Lambda(loss_module.Normalization_layer)([opt_heatmap, sar_heatmap, xcorr])

        # Crop the Normalized Cross Correlation heatmap so that matches correspond to origin (top-left corner) of the template
        out = Cropping2D(cropping=response_crop, data_format="channels_first")(xcorr)

        # Move channels back to inital position
        out = Permute((2, 3, 1))(out)

        # Averagely reduce the channel number to 1, sharpen output if normalized
        out = Lambda(lambda x: tf.divide(tf.reduce_mean(x, axis=3, keepdims=True), 1 / 30, name="normalize"))(
                out)  # 1/30 is a temperature factor




        model = Model(inputs=[opt_in, sar_in], outputs=out)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss=loss)

        # return model
        self.model = model
        model.summary()  # 这行会打印模型的总结信息
        return model

    def export_model_architecture(self):
        """ Plot model architecture (pdf is also possible)"""
        if self.model is not None:
            tf.keras.utils.plot_model(self.model, to_file=self.model_name + '.png', show_shapes=True,
                                      show_layer_names=True)

    def train(self, training_data: tf.data.Dataset, validation_data: tf.data.Dataset, epochs=5):
        if utils.confirmCommand("Train a new model?"):
            print("--training")
            def on_epoch_end(epoch,logs):
                heatmaps = self.predict_heatmap(validation_data)
                error = SarOptMatch.evaluation.print_results(validation_data, heatmaps)
            my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1),  # 15
                            # tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                 factor=0.5,
                                                                 patience=3,
                                                                 min_lr=0.00001,
                                                                 verbose=1),
                            LambdaCallback(on_epoch_end=on_epoch_end)]

            history = self.model.fit(training_data, epochs=epochs,
                                     validation_data=validation_data,
                                     callbacks=my_callbacks)

            # print("--Saving weights")
            # self.model.save_weights("weights/" + self.model_name + ".h5")
            # print("--Saving history")
            # with open("weights/" + self.model_name + "_history", 'wb') as file:
            #     pickle.dump(history.history, file)

            # self.export_attributes()

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

    # def calculate_features(self, validation_data: tf.data.Dataset):
    #     if self.model is not None:
    #         print("--Calculating feature maps")
    #         layer_names = [layer.name for layer in self.model.layers]
    #
    #         # 确定需要的输出层
    #         feature_layer_names = ['psi_opt_o', 'psi_SAR_o'] + (['psi_opt_d', 'psi_sar_d'] if self.multiscale else [])
    #         print("Features:", ', '.join(feature_layer_names))
    #
    #         # 获取特征层的输出
    #         feature_maps = [self.model.get_layer(name).output for name in feature_layer_names if name in layer_names]
    #
    #         # 检查所有输入是否被包含
    #         all_inputs = self.model.inputs
    #         print("Model inputs:", all_inputs)
    #
    #         # 创建新的可视化模型
    #         try:
    #             visualization_model = tf.keras.Model(inputs=all_inputs, outputs=feature_maps)
    #         except ValueError as e:
    #             print("Error in creating model:", e)
    #             # 如果出现错误，可能需要更详细地检查每个输入的连接方式
    #             raise
    #
    #         # 使用新模型预测特征映射
    #         feature_maps = visualization_model.predict(validation_data)
    #         return feature_maps
    #     else:
    #         raise ValueError("Model not defined")

    def calculate_features(self, validation_data: tf.data.Dataset):
        """
        Create a model which output are the feature maps.
        'matcher' is an instance of the <SarOptMatch.architectures.SAR_opt_Matcher> class
        """

        if self.model is not None:
            print("--Calculating feature maps")
            # Scan the matcher.model to find the right layers
            layer_names = [layer.name for layer in self.model.layers]

            psi_opt_o = layer_names.index("psi_opt_o")
            psi_sar_o = layer_names.index("psi_SAR_o")
            feature_maps = [self.model.layers[psi_opt_o].output, self.model.layers[psi_sar_o].output]
            if not self.multiscale:
                print("Features: psi_opt_o and psi_SAR_o")
            if self.multiscale:
                psi_opt_d = layer_names.index("psi_opt_d")
                psi_sar_d = layer_names.index("psi_SAR_d")
                feature_maps = [self.model.layers[psi_opt_o].output, self.model.layers[psi_sar_o].output,
                                self.model.layers[psi_opt_d].output, self.model.layers[psi_sar_d].output]
                print("Features: psi_opt_o, psi_SAR_o, psi_opt_d,  psi_SAR_d")
            visualization_model = tf.keras.Model(inputs=self.model.inputs, outputs=feature_maps)

            feature_maps = visualization_model.predict(validation_data)
            return feature_maps
        else:
            raise ("Model not defined")






 # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""StyleGAN论文里使用的模型架构。"""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# 用于操纵4D激活张量的原始操作。
# 这些梯度不是必须有效的，甚至不必有意义。

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # 处理滤波器核f。
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[1]), 1])

    # 无操作=>提前退出。
    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    # 使用depthwise_conv2d卷积。
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d()不支持fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, 1, stride, stride]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NCHW')  
    # 这个depthwise_conv2d卷积挺有意思。普通的卷积，我们对卷积核每一个out_channel的两个通道分别和输入的两个通道做卷积相加，得到feature map的一个channel，
    # 而depthwise_conv2d卷积，我们对每一个对应的in_channel，分别卷积生成两个out_channel，
    # 所以获得的feature map的通道数量可以用in_channel* channel_multiplier来表达，这个channel_multiplier，
    # 就可以理解为卷积核的第四维。参见博客：https://blog.csdn.net/mao_xiao_feng/article/details/78003476。
    x = tf.cast(x, orig_dtype)
    return x

def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 运用增益值。
    if gain != 1:
        x *= gain

    # 无操作=>提前退出。
    if factor == 1:
        return x

    # 上采用使用tf.tile()，下面是骚操作。
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])  # 把特征的高维度上的每个像素再封装一层，特征的宽维度上的每个像素再封装一层
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])  # 像素copy一下
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])  # 整合进原始维度下，相当于高和宽放大了factor倍，扩展的像素值是一样的
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 卷积核大小为2x2 => 下采用采用_blur2d()
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # 运用增益值。
    if gain != 1:
        x *= gain

    # 无操作=>提前退出。
    if factor == 1:
        return x

    # 下采样时采用平均池化。
    # 注意：需要tf_config ['graph_options.place_pruned_graph'] = True 才能正常工作。
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# 用于操纵4D激活张量的高级操作。
# 这些梯度的含义需要尽可能有效（使用了@tf.custom_gradient装饰器，表明函数定义中给出了函数梯度的定义）。

def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):  # 定义一个函数，返回模糊后的特征值以及该函数的梯度计算式grad()
            y = _blur2d(x, f, normalize)  # 模糊后的特征值y
            @tf.custom_gradient
            def grad(dy):  # func函数的梯度计算式
                dx = _blur2d(dy, f, normalize, flip=True)  # d_func(x)/d_x=_blur2d(d_x)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)  
                # func的梯度(即grad)用_blur2d()去近似，grad的梯度也用_blur2d()去近似(对于一阶导和二阶导都用_blur2d()作近似，可能是因为_blur2d()函数的结果(模糊)等效为目的(变模糊))。
            return y, grad
        return func(x)

def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):  # 定义一个函数，返回上采样后的特征值以及该函数的梯度计算式grad()
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):  # func函数的梯度计算式与blur2d()写法类似，对于此种函数具有通用性
                dx = _downscale2d(dy, factor, gain=factor**2)  # 上采样的增益值取放大倍数的平方
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
        return func(x)

def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):  # 定义一个函数，返回下采样后的特征值以及该函数的梯度计算式grad()
            y = _downscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):  # func函数的梯度计算式与blur2d()写法类似，对于此种函数具有通用性
                dx = _upscale2d(dy, factor, gain=1/factor**2)  # 下采样的增益值取放大倍数平方的倒数
                return dx, lambda ddx: _downscale2d(ddx, factor)
            return y, grad
        return func(x)

#----------------------------------------------------------------------------
# 获取/创建卷积层或完全连接层的权重张量，采用He的初始化方法。

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1])  # fan_in表示某卷积核参数个数(h*w*fmaps_in)或dense层输入节点数目fmaps_in
    # np.prod 给定维度上各个元素的乘积
    he_std = gain / np.sqrt(fan_in)  # He公式为：0.5*n*var(w)=1 , so：std(w)=sqrt(2)/sqrt(n)=gain/sqrt(fan_in)

    # 均衡学习率以及自定义学习率变化率。
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul  # 获得scale后的归一化值
        runtime_coef = lrmul  # 网络也实时scale

    # 创建变量。
    init = tf.initializers.random_normal(0, init_std)  # He的初始化方法能够确保网络初始化的时候，随机初始化的参数不会大幅度地改变输入信号的强度。
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef  # StyleGAN中不仅限初始状态scale而且是实时scale

#----------------------------------------------------------------------------
# 全连接层

def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])  
        # 如果x的维度超过2，把高于2的维度的值连乘起来作为第二维度的值？？
    w = get_weight([x.shape[1].value, fmaps], **kwargs)  # 创建一个[512,512]的全连接层
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# 卷积层

def conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')  # 简单的全卷积

#----------------------------------------------------------------------------
# 融合卷积+缩放。
# 与单独执行操作相比，速度更快且使用的内存更少。

def upscale2d_conv2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 128  # x的高和宽≥64的话使用融合，否则不使用融合

    # 不融合=>直接调用各个操作。
    if not fused_scale:
        return conv2d(upscale2d(x), fmaps, kernel, **kwargs)

    # 融合=>使用tf.nn.conv2d_transpose()同时执行两个操作。
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)  # 创建一个卷积层w，shape为[kernel, kernel, fmaps_in, fmaps_out]
    w = tf.transpose(w, [0, 1, 3, 2])  # 将w转成[kernel, kernel, fmaps_out, fmaps_in]，后面tf函数中卷积核输出通道在前（tf坑太多。。）
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')  # 对w进行填充，在两个kernel的第0维和最后一维分别填充0，w的shape变为[kernel+2, kernel+2, fmaps_in, fmaps_out]
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])  # 填充区域求和两次，非填充区域求和四次，w的shape不变。为什么要对卷积核做这样的处理呢？？
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]  # 即output_shape，定义反卷积需输出的维度（高宽扩大两倍）
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')  
    # 最终的反卷积输出，可以验算一下：特征x的维度是[Num,fmaps_in,H,W],卷积核w的维度是[kernel+2, kernel+2, fmaps_out, fmaps_in],
    # strides为[1,1,2,2]，这三者做反卷积的输出维度就是[Num,fmaps_out,H×2,W×2]，刚好就是os的shape。

def conv2d_downscale2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) >= 128

    # 不融合=>直接调用各个操作。
    if not fused_scale:
        return downscale2d(conv2d(x, fmaps, kernel, **kwargs))

    # 融合=>使用tf.nn.conv2d_transpose()同时执行两个操作。
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)  
    # 创建一个卷积层w，shape为[kernel, kernel, fmaps_in, fmaps_out]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')  
    # 对w进行填充，在两个kernel的第0维和最后一维分别填充0，w的shape变为[kernel+2, kernel+2, fmaps_in, fmaps_out]
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25  
    # 填充区域求和两次，非填充区域求和四次，最后再除以4，w的shape不变。为什么要对卷积核做这样的处理呢？？
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')  
    # 最终的卷积输出，可以验算一下：特征x的维度是[Num,fmaps_in,H,W],卷积核w的维度是[kernel+2, kernel+2, fmaps_in, fmaps_out],strides为[1,1,2,2]，这三者做卷积的输出维度是[Num,fmaps_out,H//2,W//2]

#----------------------------------------------------------------------------
# 对给定的激活张量施加偏差。

def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])   # 偏差应该是只对于第二维有效？？（Dense层中已把高于2的维度的值连乘起来作为第二维度的值）

#----------------------------------------------------------------------------
# Leaky ReLU激活函数。下面这个比tf.nn.leaky_relu()更有效，并且支持FP16。

def leaky_relu(x, alpha=0.2):
    with tf.variable_scope('LeakyReLU'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')  # 定义常数α
        @tf.custom_gradient
        def func(x):
            y = tf.maximum(x, x * alpha)  # 返回x和αx中的最大值，折线的形状
            @tf.custom_gradient
            def grad(dy):
                dx = tf.where(y >= 0, dy, dy * alpha)  # 自定义1阶导，使得leaky_relu全域可求导
                return dx, lambda ddx: tf.where(y >= 0, ddx, ddx * alpha)  # 2阶导也定义为1或α
            return y, grad
        return func(x)

#----------------------------------------------------------------------------
# 逐像素特征向量归一化。

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)  
        # x = x/√(x2_avg+ε)
        # pixel norm 沿channel维度做归一化(axis=1)

#----------------------------------------------------------------------------
# 实例规一化。

def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4  # NCHW
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2,3], keepdims=True)  # 实例归一化仅对HW做归一化，所以axis是2,3
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2,3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)  # x = (x-x_mean)/√(x2_avg+ε)
        return x

#----------------------------------------------------------------------------
# 样式调制（AdaIN）

def style_mod(x, dlatent, **kwargs):
    with tf.variable_scope('StyleMod'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[1]*2, gain=1, **kwargs))  # 仿射变换A（通过全连接层将dlatent扩大一倍）
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))  
        # 扩大后的dlatent转换为放缩因子y_s,i和偏置因子y_b,i ？？？
        return x * (style[:,0] + 1) + style[:,1]  # 对卷积后的x执行样式调制（自适应实例归一化）

#----------------------------------------------------------------------------
# 噪音输入。

def apply_noise(x, noise_var=None, randomize_noise=True):
    assert len(x.shape) == 4  # NCHW
    with tf.variable_scope('Noise'):
        if noise_var is None or randomize_noise:  # 添加随机噪音
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:  # 添加指定噪音
            noise = tf.cast(noise_var, x.dtype)
        weight = tf.get_variable('weight', shape=[x.shape[1].value], initializer=tf.initializers.zeros())  # 噪音的权重
        return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])

#----------------------------------------------------------------------------
# 小批量标准偏差。

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # 小批量一定能被group_size整除（或小于group_size）。
        s = x.shape                                             # [NCHW]  输入shape.
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   
        # [GMncHW] 将小批量拆分为M个大小为G的组。将通道拆分为n个c个通道的组。
        y = tf.cast(y, tf.float32)                              # [GMncHW] 转换成FP32。
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] 按组减去均值。
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  按组计算方差。
        y = tf.sqrt(y + 1e-8)                                   # [MncHW]  按组计算标准方差。
        y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  在特征图和像素上采取平均值。
        y = tf.reduce_mean(y, axis=[2])                         # [Mn11] 将通道划分为c个通道组。
        y = tf.cast(y, x.dtype)                                 # [Mn11]  转换回原始的数据类型。
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  按组和像素进行复制。
        return tf.concat([x, y], axis=1)                        # [NCHW]  添加为新的特征图。

#----------------------------------------------------------------------------
# StyleGAN 论文中基于Style的生成器架构.
# 生成器由两个子网络组成 (G_mapping 和 G_synthesis) 在下面定义。

def G_style(
    latents_in,                                     # 第一个输入：Z码向量 [minibatch, latent_size].
    labels_in,                                      # 第二个输入：条件标签 [minibatch, label_size].
    truncation_psi          = 0.7,                  # 截断技巧的样式强度乘数。 None = disable.
    truncation_cutoff       = 8,                    # 要应用截断技巧的层数。 None = disable.
    truncation_psi_val      = None,                 # 验证期间要使用的truncation_psi的值。
    truncation_cutoff_val   = None,                 # 验证期间要使用的truncation_cutoff的值。
    dlatent_avg_beta        = 0.995,                # 在训练期间跟踪W的移动平均值的衰减率。 None = disable.
    style_mixing_prob       = 0.9,                  # 训练期间混合样式的概率。 None = disable.
    is_training             = False,                # 网络正在接受训练？ 这个选择可以启用和禁用特定特征。
    is_validation           = False,                # 网络正在验证中？ 这个选择用于确定truncation_psi的值。
    is_template_graph       = False,                # True表示由Network类构造的模板图，False表示实际评估。
    components              = dnnlib.EasyDict(),    # 子网络的容器。调用时候保留。
    **kwargs):                                      # 子网络的参数们 (G_mapping 和 G_synthesis)。

    # 参数验证。
    assert not is_training or not is_validation        # 不能同时出现训练/验证状态
    assert isinstance(components, dnnlib.EasyDict)     # components作为EasyDict类，后续被用来装下synthesis和mapping两个网络
    if is_validation:    # 把验证期间要使用的truncation_psi_val和truncation_cutoff_val值赋过来（默认是None），也就是验证期不使用截断
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None  # 训练期间或截断率为1时，不使用截断
    if is_training or (truncation_cutoff is not None and not tflib.is_tf_expression(truncation_cutoff) and truncation_cutoff <= 0):
        truncation_cutoff = None  # 训练期间或截断层为0时，不使用截断
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None  # 非训练期间或计算平均W时的衰减率为1时，不使用衰减
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None  # 非训练期间或样式混合的概率小于等于0时，不使用样式混合

    # 设置子网络。
    if 'synthesis' not in components:  # 载入合成网络
        components.synthesis = tflib.Network('G_synthesis', func_name=G_synthesis, **kwargs)
    num_layers = components.synthesis.input_shape[1]  # num_layers = 18
    dlatent_size = components.synthesis.input_shape[2]  # dlatent_size = (18,512)
    if 'mapping' not in components:  # 载入映射网络
        components.mapping = tflib.Network('G_mapping', func_name=G_mapping, dlatent_broadcast=num_layers, **kwargs)

    # 设置变量。
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)  
    # 初始化为0。lod的定义式为：lod = resolution_log2 - res，其中resolution_log2（=10）表示最终分辨率级别，res表示当前层对应的分辨率级别（2-10）。
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)  # 人脸平均值

    # 计算映射网络输出。
    dlatents = components.mapping.get_output_for(latents_in, labels_in, **kwargs)

    # 更新W的移动平均值。
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)   # 找到新batch的dlatent平均值
            # ？？？
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta)) 
            # 把batch的dlatent平均值朝着总dlatent平均值以dlatent_avg_beta步幅靠近，作为新的人脸dlatent平均值
            # update_op 是一个用于sess的变量
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)  # 确保update_op操作完成
            # with tf.control_dependencies: 在with包含的操作operation执行前先执行op列表即[update_up]:
            # tf.identity(x)是一个operation。所以会确保update_op操作完成。
            # tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中

    # 执行样式混合正则化。
    if style_mixing_prob is not None:
        with tf.name_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, **kwargs)  # 用来做样式混合的随机中间向量
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]   # 层的索引：[[[0],[1],[2],[3],[4]...[17]]]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2  # 当前层等于总层数减去lod_in的两倍,因为每个分辨率对应两层
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,   # 如果随机值小于样式混合的概率，则从1到当前层随机选一个层，否则保留原层
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)  
            # 对于1到mixing_cutoff层保留dlatents的值，其余层采用dlatents2的值替换

    # 应用截断技巧。
    if truncation_psi is not None and truncation_cutoff is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]  # 层的索引：[[[0],[1],[2],[3],[4]...[17]]]
            ones = np.ones(layer_idx.shape, dtype=np.float32)  # ones:[[[1],[1],[1],[1],[1]...[1]]]
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)  
            # 截断的步幅，需要截断的层步幅为truncation_psi，否则为1
            dlatents = tflib.lerp(dlatent_avg, dlatents, coefs)  
            # 截断，用平均脸dlatent_avg朝着当前脸dlatents以coefs步幅靠近，最后得到的结果取代当前脸dlatents

    # 计算合成网络输出。
    with tf.control_dependencies([tf.assign(components.synthesis.find_var('lod'), lod_in)]):
        images_out = components.synthesis.get_output_for(dlatents, force_clean_graph=is_template_graph, **kwargs)
    return tf.identity(images_out, name='images_out')  # 返回生成的图片

#----------------------------------------------------------------------------
# StyleGAN论文中使用的映射网络

def G_mapping(
    latents_in,                             # 第一个输入：Z码向量 [minibatch, latent_size].
    labels_in,                              # 第二个输入：条件标签 [minibatch, label_size].
    latent_size             = 512,          # 潜在向量（Z）维度。
    label_size              = 0,            # 标签尺寸，0表示没有标签。
    dlatent_size            = 512,          # 解缠后的中间向量 (W) 维度。
    dlatent_broadcast       = None,         # 将解缠后的中间向量（W）输出为[minibatch，dlatent_size]或[minibatch，dlatent_broadcast，dlatent_size]格式。
    mapping_layers          = 8,            # 映射网络的层数。
    mapping_fmaps           = 512,          # 映射层中的特征图维度。
    mapping_lrmul           = 0.01,         # 映射层的学习率变化率。
    mapping_nonlinearity    = 'lrelu',      # 激活函数: 'relu', 'lrelu'.
    use_wscale              = True,         # 启用均等的学习率？
    normalize_latents       = True,         # 在将潜在向量（Z）馈送到映射层之前对其进行归一化？
    dtype                   = 'float32',    # 用于激活和输出的数据类型。
    **_kwargs):                             # 忽略无法识别的关键字参数。

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[mapping_nonlinearity]

    # 输入，处理好latent的大小和格式，其值赋给x，用x来标识网络的输入。
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # 嵌入标签并将其与潜码连接起来。
    if label_size:  # 原始StyleGAN是无条件训练集，这部分不用考虑
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # 归一化潜码。
    if normalize_latents:
        x = pixel_norm(x)

    # 映射层。
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps  
            # 除去最后一层输出维度是中间向量的维度dlatent_size，其他层输出都是映射特征图的维度mapping_fmaps
            x = dense(x, fmaps=fmaps, gain=gain, use_wscale=use_wscale, lrmul=mapping_lrmul)  # 全连接层
            x = apply_bias(x, lrmul=mapping_lrmul)  # 添加偏置
            x = act(x)  # 添加激活函数

    # 广播。
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):   
        # 这儿只定义了简单的复制扩充，广播前x的维度是(?,512),广播后x的维度是(?,18,512)
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # 输出。
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

#----------------------------------------------------------------------------
# StyleGAN论文中使用的合成网络。

def G_synthesis(
    dlatents_in,                        # 输入：解缠的中间向量 (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # 解缠的中间向量 (W) 的维度。
    num_channels        = 3,            # 输出颜色通道数。
    resolution          = 1024,         # 输出分辨率。
    fmap_base           = 8192,         # 特征图的总数目，这儿取8192因为512*(18-2)=8192。   
    fmap_decay          = 1.0,          # 当分辨率翻倍时以log2降低特征图，这儿指示降低的速率。
    fmap_max            = 512,          # 在任何层中特征图的最大数量。
    use_styles          = True,         # 启用样式输入
    const_input_layer   = True,         # 第一层是常数？
    use_noise           = True,         # 启用噪音输入？
    randomize_noise     = True,         # True表示每次都随机化噪声输入（不确定），False表示从变量中读取噪声输入。
    nonlinearity        = 'lrelu',      # 激活函数: 'relu', 'lrelu'
    use_wscale          = True,         # 启用均等的学习率？
    use_pixel_norm      = False,        # 启用逐像素特征向量归一化？
    use_instance_norm   = True,         # 启用实例规一化？
    dtype               = 'float32',    # 用于激活和输出的数据类型。
    fused_scale         = 'auto',       # True = 融合卷积+缩放，False = 单独操作，'auto'= 自动决定。
    blur_filter         = [1,2,1],      # 重采样激活时应用的低通卷积核（Low-pass filter）。None表示不过滤。
    structure           = 'auto',       # 'fixed' = 无渐进式增长，'linear' = 人类可读，'recursive' = 有效，'auto' = 自动选择。
    is_template_graph   = False,        # True表示由Network类构造的模板图，False表示实际评估。
    force_clean_graph   = False,        # True表示构建一个在TensorBoard中看起来很漂亮的干净图形，False表示默认设置。
    **_kwargs):                         # 忽略无法识别的关键字参数。

    resolution_log2 = int(np.log2(resolution))  # 计算分辨率是2的多少次方
    assert resolution == 2**resolution_log2 and resolution >= 4   # 分辨率需要大于等于32，因为训练从学习生成32*32的图片开始
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)  
    # nf()返回在第stage层中特征图的数量————当stage<=4时，特征图数量为512；当stage>4时，每多一层特征图数量就减半。
    def blur(x): return blur2d(x, blur_filter) if blur_filter else x  # 对图片进行滤波模糊操作，有利于降噪
    if is_template_graph: force_clean_graph = True
    if force_clean_graph: randomize_noise = False
    if structure == 'auto': structure = 'linear' if force_clean_graph else 'recursive'  
    # 依据force_clean_graph选择架构为'linear'或'recursive'
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]  # 激活函数
    num_layers = resolution_log2 * 2 - 2  # 因为每个分辨率有两层，所以层数为：分辨率级别(10)*2-2=18
    num_styles = num_layers if use_styles else 1  # 样式层数
    images_out = None

    # 主要输入。
    dlatents_in.set_shape([None, num_styles, dlatent_size])  # dlatents_in是通过广播得到的中间向量，维度是(?,18,512)
    dlatents_in = tf.cast(dlatents_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)  # lod_in是一个指定当前输入分辨率级别的参数，规定lod = resolution_log2 - res

    # 创建噪音。
    noise_inputs = []
    if use_noise:
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2  # [2,2,3,3,…,10,10]
            shape = [1, use_noise, 2**res, 2**res]  # 不同层的噪音shape从[1,1,4,4]一直到[1,1,1024,1024]
            noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))  # 随机初始化噪音

    # ★每一层最后需要做的事情。
    def layer_epilogue(x, layer_idx):
        if use_noise:
            x = apply_noise(x, noise_inputs[layer_idx], randomize_noise=randomize_noise)  # 应用噪音
        x = apply_bias(x)  # 应用偏置
        x = act(x)  # 应用激活函数
        if use_pixel_norm:
            x = pixel_norm(x)  # 逐像素归一化
        if use_instance_norm:
            x = instance_norm(x)  # 实例归一化
        if use_styles:
            x = style_mod(x, dlatents_in[:, layer_idx], use_wscale=use_wscale)  # 样式调制，AdaIN
        return x

    # 早期的层。
    with tf.variable_scope('4x4'):
        if const_input_layer:  # 合成网络的起点是否为固定常数，StyleGAN中选用固定常数。
            with tf.variable_scope('Const'):
                x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.ones())  # 初始为常数变量，shape为(1,512,4,4)
                x = layer_epilogue(tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1]), 0)  # 第0层的层末调制
        else:
            with tf.variable_scope('Dense'):
                x = dense(dlatents_in[:, 0], fmaps=nf(1)*16, gain=gain/4, use_wscale=use_wscale)  # 调整增益值以匹配ProGAN的官方实现（ProGAN的初始起点不是常数，而就是latent）
                x = layer_epilogue(tf.reshape(x, [-1, nf(1), 4, 4]), 0)
        with tf.variable_scope('Conv'):
            x = layer_epilogue(conv2d(x, fmaps=nf(1), kernel=3, gain=gain, use_wscale=use_wscale), 1)  # 第1层为卷积层，添加层末调制

    # 为剩余层构建block块。
    def block(res, x):  # res从3增加到resolution_log2；这些层被写在函数里方便网络需要时再创建。
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0_up'):  # 第2,4,6…,16层为上采样层；上采样之后会加一个模糊滤波以降噪。
                x = layer_epilogue(blur(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)), res*2-4)
            with tf.variable_scope('Conv1'):  # 第3,5,7…,17层为卷积层
                x = layer_epilogue(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale), res*2-3)
            return x
    def torgb(res, x): # res从2增加到resolution_log2；这个函数实现特征图到RGB图像的转换。
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))  # ToRGB是通过一个简单卷积实现的

    # 固定结构：简单高效，但不支持渐进式增长。
    if structure == 'fixed':
        for res in range(3, resolution_log2 + 1):  # res从3增加到resolution_log2
            x = block(res, x)  # 相当于直接构建了一个1024*1024分辨率的生成器网络
        images_out = torgb(resolution_log2, x)

    # ★线性结构：简单但效率低下。
    if structure == 'linear':
        images_out = torgb(2, x)
        for res in range(3, resolution_log2 + 1):  # res从3增加到resolution_log2
            lod = resolution_log2 - res
            x = block(res, x)
            img = torgb(res, x)
            images_out = upscale2d(images_out)  # 通过upscale2d()构建上采样层，将当前分辨率放大一倍
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = tflib.lerp_clip(img, images_out, lod_in - lod)  
                # 依靠含大小值裁剪的线性插值实现图片放大，相当于在过渡阶段实现平滑过渡

    # ★递归结构：复杂但高效。
    # lambda: 匿名函数
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)  
            # 返回一个函数，依据是否满足new_cond决定返回new_lambda函数还是cur_lambda函数
        def grow(x, res, lod):
            y = block(res, x)
            img = lambda: upscale2d(torgb(res, y), 2**lod)
            img = cset(img, (lod_in > lod), lambda: upscale2d(tflib.lerp(torgb(res, y), upscale2d(torgb(res - 1, x)), lod_in - lod), 2**lod))  
            # 如果输入层数lod_in超过当前层lod的话（但同时小于lod+1），实现从lod对应分辨率到lod_in对应分辨率的扩增，采用线性插值；否则按lod处理。
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))  
            # 如果lod_in小于lod且不是最后一层的话（也就是前者的res超过后者的res），表明可以进入到下一级分辨率上了，此时res+1, lod-1
            return img()
        images_out = grow(x, 3, resolution_log2 - 3)  
        # res一开始为3，lod一开始为resolution_log2 - res，利用递归就可以构建res从3增加到resolution_log2的全部架构

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')  # 输出

#----------------------------------------------------------------------------
# StyleGAN论文中使用的判别器结构。

def D_basic(
    images_in,                          # 第一个输入：图片 [minibatch, channel, height, width].
    labels_in,                          # 第二个输入：标签 [minibatch, label_size].
    num_channels        = 1,            # 输入颜色通道数。 根据数据集覆盖。
    resolution          = 32,           # 输入分辨率。 根据数据集覆盖。
    label_size          = 0,            # 标签的维数，0表示没有标签。根据数据集覆盖。
    fmap_base           = 8192,         # 特征图的总数目，这儿取8192因为512*(18-2)=8192。
    fmap_decay          = 1.0,          # 当分辨率翻倍时以log2降低特征图，这儿指示降低的速率。
    fmap_max            = 512,          # 在任何层中特征图的最大数量。
    nonlinearity        = 'lrelu',      # 激活函数: 'relu', 'lrelu'。
    use_wscale          = True,         # 启用均等的学习率？
    mbstd_group_size    = 4,            # 小批量标准偏差层的组大小，0表示禁用。
    mbstd_num_features  = 1,            # 小批量标准偏差层的特征数量。
    dtype               = 'float32',    # 用于激活和输出的数据类型。
    fused_scale         = 'auto',       # True = 融合卷积+缩放，False = 单独操作，'auto'= 自动决定。
    blur_filter         = [1,2,1],      # 重采样激活时应用的低通卷积核（Low-pass filter）。None表示不过滤。
    structure           = 'auto',       # 'fixed' = 无渐进式增长，'linear' = 人类可读，'recursive' = 有效，'auto' = 自动选择。
    is_template_graph   = False,        # True表示由Network类构造的模板图，False表示实际评估。
    **_kwargs):                         # 忽略无法识别的关键字参数。

    resolution_log2 = int(np.log2(resolution))  # 计算分辨率是2的多少次方
    assert resolution == 2**resolution_log2 and resolution >= 4  # 分辨率需要大于等于32，因为训练从学习生成32*32的图片开始
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)  # nf()返回在第stage层中特征图的数量————当stage<=4时，特征图数量为512；当stage>4时，每多一层特征图数量就减半。
    def blur(x): return blur2d(x, blur_filter) if blur_filter else x  # 对图片进行滤波模糊操作，有利于降噪
    if structure == 'auto': structure = 'linear' if is_template_graph else 'recursive'  # 依据is_template_graph选择架构为'linear'或'recursive'
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]  # 激活函数
    # 输入处理
    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)  # 输入的分辨率级别, lod = resolution_log2 - res
    scores_out = None  # 输出分数

    # 构建block块。
    def fromrgb(x, res):  # res从2增加到resolution_log2；这个函数实现RGB图像到特征图的转换。
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))  
            # 简单卷积实现，并应用激活函数
    def block(x, res):  # res从2增加到resolution_log2；这些层被写在函数里方便网络需要时再创建。
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3:  # 8x8分辨率及以上
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))  # 构建一个卷积层
                with tf.variable_scope('Conv1_down'):
                    x = act(apply_bias(conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))  # 构建一个下采样层
            else:  # 4x4分辨率，得到判别分数scores_out
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)  # 构建一个小偏量标准偏差层
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))  # 卷积
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))  # 全连接
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=max(label_size, 1), gain=1, use_wscale=use_wscale))  # 全连接
            return x

    # 固定结构：简单高效，但不支持渐进式增长。
    if structure == 'fixed':
        x = fromrgb(images_in, resolution_log2)  # 将输入图片转换为特征x
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)  # 相当于直接构建了一个从1024*1024分辨率降到4*4分辨率的下采样网络
        scores_out = block(x, 2)  # 输出为判别分数

    # 线性结构：简单但效率低下。
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)  # 将输入图片转换为特征x
        for res in range(resolution_log2, 2, -1):  # res从resolution_log2降低到3
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)  # 通过downscale2d()构建下采样层，将当前分辨率缩小一倍
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = tflib.lerp_clip(x, y, lod_in - lod)  # 依靠含大小值裁剪的线性插值实现图片缩小，相当于在过渡阶段实现平滑过渡
        scores_out = block(x, 2)

    # 递归结构：复杂但高效。
    if structure == 'recursive':  
    # 注意判别器在训练时是输入图片先进入lod最小的层，但是构建判别网络时是lod从大往小构建，所以递归的过程是与生成器相反的。
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)  
            # 返回一个函数，依据是否满足new_cond决定返回new_lambda函数还是cur_lambda函数
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)  # 先暂时将下采样函数赋给x
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))  
            # 非第一层时，如果输入层数lod_in小于当前层lod的话，表明可以进入到下一级分辨率上了，将grow()赋给x；否则x还是保留为下采样函数。
            x = block(x(), res); y = lambda: x  # x执行一次自身的函数，构建出一个block，并将结果赋给y（以函数的形式）
            if res > 2: y = cset(y, (lod_in > lod), lambda: tflib.lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))  # 非最后一层时，如果输入层数lod_in大于当前层lod的话，表明需要进行插值操作，将lerp()赋给y；否则y还是保留为之前的操作。
            return y()
        scores_out = grow(2, resolution_log2 - 2)  # 构建判别网络时是lod从大往小构建，所以一开始的lod输入为8

    # 标签条件来自“哪种GAN训练方法实际上会收敛？”
    if label_size:
        with tf.variable_scope('LabelSwitch'):
            scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)

    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out  # 输出

#----------------------------------------------------------------------------
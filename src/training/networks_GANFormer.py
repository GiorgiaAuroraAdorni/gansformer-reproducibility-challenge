﻿# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Network architectures used in the StyleGAN2 paper."""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act
import math

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.
# Shape operations
# ----------------------------------------------------------------------------

# Return the shape of tensor as a list, preferring static dimensions when available
def get_shape(x):
    shape, dyn_shape = x.shape.as_list().copy(), tf.shape(x)
    for index, dim in enumerate(shape):
        if dim is None:
            shape[index] = dyn_shape[index]
    return shape
# Produce trainable embeddings of shape [size, dim], uniformly/normally initialized
def get_embeddings(size, dim, init = "uniform", name = None):
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    with tf.variable_scope(name):
        emb = tf.get_variable(name = "emb", shape = [size, dim], initializer = initializer)
    return emb
# Flatten all dimensions of a tensor except the fist/last one
def to_2d(x, mode):
    shape = get_shape(x)
    if len(shape) == 2:
        return x
    if mode == "last":
        return tf.reshape(x, [-1, shape[-1]])
    else:
        return tf.reshape(x, [shape[0], np.prod(get_shape(x)[1:])])
#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight',  name = None):
    if name is not None:
        weight_var = "{}_{}".format(weight_var, name)
    
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolution layer with optional upsampling or downsampling.

def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
    return x

#----------------------------------------------------------------------------
# Apply bias and activation func.

def apply_bias_act(x, act='linear', alpha=None, gain=None, lrmul=1, bias_var='bias', name = None):
    if name is not None:
        bias_var = "{}_{}".format(bias_var, name)
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain)
# Dropout and masking
# ----------------------------------------------------------------------------

# Create a random mask of a chosen shape and with probability 'dropout' to be dropped (=0)
def random_dp_binary(shape, dropout):
    if dropout == 0:
        return tf.ones(shape)
    eps = tf.random.uniform(shape)
    keep_mask = (eps >= dropout)
    return keep_mask
# Perform dropout
def dropout(x, dp, noise_shape = None):
    if dp is None or dp == 0.0:
        return x
    return tf.nn.dropout(x, keep_prob = 1.0 - dp, noise_shape = noise_shape)

# Set a mask for logits (set -Inf where mask is 0)
def logits_mask(x, mask):
    return x + tf.cast(1 - tf.cast(mask, tf.int32), tf.float32) * -10000.0
#----------------------------------------------------------------------------
# Naive upsampling (nearest neighbor) and downsampling (average pooling).

def naive_upsample_2d(x, factor=2):
    with tf.variable_scope('NaiveUpsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H, 1, W, 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        return tf.reshape(x, [-1, C, H * factor, W * factor])

def naive_downsample_2d(x, factor=2):
    with tf.variable_scope('NaiveDownsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H // factor, factor, W // factor, factor])
        return tf.reduce_mean(x, axis=[3,5])

#----------------------------------------------------------------------------
# Modulated convolution layer.

def modulated_conv2d_layer(x, 
    y, 
    fmaps, 
    kernel, 
    up=False, 
    down=False, 
    demodulate=True, 
    resample_kernel=None, 
    gain=1, 
    use_wscale=True, 
    lrmul=1, 
    fused_modconv=True, 
    weight_var='weight', 
    mod_weight_var='mod_weight', 
    mod_bias_var='mod_bias'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    s = dense_layer(y, fmaps=x.shape[1].value, weight_var=mod_weight_var) # [BI] Transform incoming W to style.
    s = apply_bias_act(s, bias_var=mod_bias_var) + 1 # [BI] Add bias (initially 1).
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

    # Convolution with optional up/downsampling.
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')

    # Reshape/scale output.
    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
    return x

#----------------------------------------------------------------------------
# Minibatch standard deviation layer.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1, have_last = True):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    last = [s[3]] if have_last else []
    y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2]] + last)   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis = [2, 3] + ([4] if have_last else []), keepdims = True) # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
    y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2]] + last)             # [NnHW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.
# Apply feature normalization, either instance, batch or layer normalization.
# x shape is NCHW
def norm(x, norm_type, parametric = True):
    if norm_type == "instance":
        x = tf.contrib.layers.instance_norm(x, data_format = "NCHW", center = parametric, scale = parametric)
    elif norm_type == "batch":
        x = tf.contrib.layers.batch_norm(x, data_format = "NCHW", center = parametric, scale = parametric)
    elif norm_type == "layer":
        x = tf.contrib.layers.layer_norm(inputs = x, begin_norm_axis = -1, begin_params_axis = -1)
    return x
# Non-Linear networks
# ----------------------------------------------------------------------------

# Non-linear layer with a resnet connection. Accept features 'x' of dimension 'dim',
# and use nonlinearty 'act'. Optionally perform attention from x to y
# (meaning information flows y -> x).
def nnlayer(x, dim, act, lrmul = 1, y = None, ff = True, pool = False, name = "", **kwargs):
    shape = get_shape(x)
    _x = x

    if y is not None:
        x = transformer_layer(from_tensor = x, to_tensor = y, fmaps = dim, name = name, **kwargs)[0]

    if ff:
        if pool:
            x = to_2d(x, "last")

        with tf.variable_scope("Dense%s_0" % name):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
        with tf.variable_scope("Dense%s_1" % name):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), lrmul = lrmul)

        if pool:
            x = tf.reshape(x, shape)

        x = tf.nn.leaky_relu(x + _x)

    return x
# Multi-layer network with 'layers_num' layers, dimension 'dim', and nonlinearity 'act'.
# Optionally use resnet connections and self-attention.
# If x dimensions are not [batch_size, dim], then first pool x, according to a pooling scheme:
# - mean: mean pooling
# - cnct: concatenate all spatial features together to one vector and compress them to dimension 'dim'
# - 2d: turn a tensor [..., dim] into [-1, dim] so to create one large batch with all the elements from the last axis.
def mlp(x, resnet, layers_num, dim, act, lrmul, pooling = "mean", transformer = False, norm_type = None, **kwargs):
    shape = get_shape(x)

    if len(get_shape(x)) > 2:
        if pooling == "cnct":
            with tf.variable_scope("Dense_pool"):
                x = apply_bias_act(dense_layer(x, dim), act = act)
        elif pooling == "batch":
            x = to_2d(x, "last")
        else:
            pool_shape = (get_shape(x)[-2], get_shape(x)[-1])
            x = tf.nn.avg_pool(x, pool_shape, pool_shape, padding = "SAME", data_format = "NCHW")
            x = to_2d(x, "first")

    if resnet:
        half_layers_num = int(layers_num / 2)
        for layer_idx in range(half_layers_num):
            y = x if transformer else None
            x = nnlayer(x, dim, act, lrmul, y = y, name = layer_idx, **kwargs)
            x = norm(x, norm_type)

        with tf.variable_scope("Dense%d" % layer_idx):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)

    else:
        for layer_idx in range(layers_num):
            with tf.variable_scope("Dense%d" % layer_idx):
                x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
                x = norm(x, norm_type)

    x = tf.reshape(x, [-1] + shape[1:-1] + [dim])
    return x
# Validate transformer input shape for from/to_tensor and reshape to 2d
def process_input(t, t_pos, t_len, name):
    shape = get_shape(t)

    # from/to_tensor should be either 2 or 3 dimensions. If it's 3, then t_len should be specified.
    if len(shape) > 3:
        print("Transformer {}_tensor has {} shape. should be up to 3 dims.".format(name, shape))
        exit()
    elif len(shape) == 3:
        batch_size, t_len, _ = shape
    else:
        if t_len is None:
            print("If {}_tensor has two dimensions, must specify {}_len.".format(name, name))
            exit()
        # Infer batch size for the 2-dims case
        batch_size = tf.cast(shape[0] / t_len, tf.int32)

    # Reshape tensors to 2d
    t = to_2d(t, "last")
    if t_pos is not None:
        t_pos = tf.tile(to_2d(t_pos, "last"), [batch_size, 1])

    return t, t_pos, shape, t_len, batch_size

# Transpose tensor to scores
def transpose_for_scores(x, batch_size, num_heads, elem_num, head_size):
    x = tf.reshape(x, [batch_size, elem_num, num_heads, head_size]) # [B, N, H, S]
    x = tf.transpose(x, [0, 2, 1, 3]) # [B, H, N, S]
    return x

# Compute attention probabilities: perform softmax on att_scores and dropout
def compute_probs(scores, dp):
    # Compute attention probabilities
    probs = tf.nn.softmax(scores) # [B, N, F, T]
    shape = get_shape(probs)
    shape[-2] = 1
    # Dropout over random cells and over random full rows (randomly don't use a 'to' element)
    probs = dropout(probs, dp / 2)
    probs = dropout(probs, dp / 2, shape)
    return probs

#### K-means (as part of Duplex Attention)
#
# Basically, given the attention scores between 'from' elements to 'to' elements, compute
# the 'to' centroids of the inferred assignments relations, as in the k-means algorithm.
#
# (Intuitively, given that the bed region will get assigned to one latent, and the chair region
# will get assigned to another latent, we will compute the centroid/mean of that region and use
# it as a representative of that region/object).
# ---------------------------------------------------------------------------------------------

# Compute relative weights of different 'from' elements for each 'to' centroid.
# Namely, compute assignments of 'from' elements to 'to' elements, by normalizing the
# attention distribution over the rows, to obtain the weight contribution of each
# 'from' element to the 'to' centroid.
#
# Returns [batch_size, num_heads, to_len, from_len] for each element in 'to'
# the relative weights of assigned 'from' elements (their weighted sum is the respective centroid)
def compute_assignments(att_probs):
    centroid_assignments = (att_probs / (tf.reduce_sum(att_probs, axis = -2, keepdims = True) + 1e-8))
    centroid_assignments = tf.transpose(centroid_assignments, [0, 1, 3, 2]) # [B, N, T, F]
    return centroid_assignments

# Given queries (function of the 'from' elements) and the centroid_assignemnts
# between 'from' and 'to' elements, compute the centroid/mean queries.
#
# Some of the code here meant to be backward compatible with the pretrained networks
# and may improve in further versions of the repository.
def compute_centroids(_queries, queries, to_from, to_len, from_len, batch_size, num_heads, 
        size_head, parametric):
    
    dim = 2 * size_head
    from_elements = tf.concat([_queries, queries - _queries], axis = -1)
    from_elements = transpose_for_scores(from_elements, batch_size, num_heads, from_len, dim) # [B, N, F, H]

    # to_from represent centroid_assignments of 'from' elements to 'to' elements
    # [batch_size, num_head, to_len, from_len]
    if to_from is not None:
        # upsample centroid_assignments from the prior generator layer
        # (where image grid dimensions were x2 smaller)
        if get_shape(to_from)[-2] < to_len:
            s = int(math.sqrt(get_shape(to_from)[-2]))
            to_from = upsample_2d(tf.reshape(to_from, [batch_size * num_heads, s, s, from_len]), factor = 2, data_format = "NHWC")
            to_from = tf.reshape(to_from, [batch_size, num_heads, to_len, from_len])

        if get_shape(to_from)[-1] < from_len:
            s = int(math.sqrt(get_shape(to_from)[-1]))
            to_from = upsample_2d(tf.reshape(to_from, [batch_size * num_heads, to_len, s, s]), factor = 2, data_format = "NCHW")
            to_from = tf.reshape(to_from, [batch_size, num_heads, to_len, from_len])

        # Given:
        # 1. Centroid assignments of 'from' elements to 'to' centroid
        # 2. 'from' elements (queries)
        # Compute the 'to' respective centroids
        to_centroids = tf.matmul(to_from, from_elements)

    # Centroids initialization
    if to_from is None or parametric:
        if parametric:
            to_centroids = tf.tile(tf.get_variable("toasgn_init", shape = [1, num_heads, to_len, dim],
                initializer = tf.initializers.random_normal()), [batch_size, 1, 1, 1])
        else:
            to_centroids = apply_bias_act(dense_layer(queries, dim * num_heads, name = "key2"), name = "key2")
            to_centroids = transpose_for_scores(to_centroids, batch_size, num_heads, dim, dim)

    return from_elements, to_centroids
# Normalization operation used in attention layers. Does not scale back features x (the image)
# with parametrized gain and bias, since these will be controlled by the additive/multiplicative
# integration of as part of the transformer layer (where the latent y will modulate the image features x)
# after x gets normalized, by controlling their scale and bias (similar to the FiLM and StyleGAN approaches).
#
# Arguments:
# - x: [batch_size * num, channels]
# - num: number of elements in the x set (e.g. number of positions WH)
# - integration: type of integration -- additive, multiplicative or both
# - norm: normalization type -- instance or layer-wise
# Returns: normalized x tensor
def att_norm(x, num, integration, norm):
    shape = get_shape(x)
    x = tf.reshape(x, [-1, num] + get_shape(x)[1:])
    x = tf.cast(x, tf.float32)

    # instance axis if norm == "instance" and channel axis if norm == "layer"
    norm_axis = 1 if norm == "instance" else 2

    if integration in ["add", "both"]:
        x -= tf.reduce_mean(x, axis = norm_axis, keepdims = True)
    if integration in ["mul", "both"]:
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = norm_axis, keepdims = True) + 1e-8)

    # return x to its original shape
    x = tf.reshape(x, shape)
    return x

# Normalizes the 'tensor' elements, and then integrate the new information from
# 'control' with 'tensor', where 'control' controls the bias/gain of 'tensor'.
# norm types: batch, instance, layers
# integration types: add, mul, both
def integrate(tensor, tensor_len, control, integration, norm):
    dim = get_shape(tensor)[-1]

    # Normalize tensor
    if norm is not None:
        tensor = att_norm(tensor, tensor_len, integration, norm)

    # Compute gain/bias
    control_dim = {"add": dim, "mul": dim, "both": 2 * dim}[integration]
    bias = gain = control = apply_bias_act(dense_layer(control, control_dim, name = "out"), name = "out")
    if integration == "both":
        gain, bias = tf.split(control, 2, axis = -1)

    # Modulate the bias/gain of 'tensor'
    if integration != "add":
        tensor *= (gain + 1)
    if integration != "mul":
        tensor += bias

    return tensor
# 2d sinusoidal embeddings [size, size, dim] with size = grid size and dim = embedding dimension
# (see "Attention is all you need" paper)
def get_sinusoidal_embeddings(size, dim):
    # Standard positional embeddings in the two spatial w,h directions
    c = tf.expand_dims(tf.to_float(tf.linspace(-1.0, 1.0, size)), axis = -1)
    i = tf.to_float(tf.range(int(dim / 4)))

    peSin = tf.sin(c / (tf.pow(10000.0, 4 * i / dim)))
    peCos = tf.cos(c / (tf.pow(10000.0, 4 * i / dim)))

    peSinX = tf.tile(tf.expand_dims(peSin, axis = 0), [size, 1, 1])
    peCosX = tf.tile(tf.expand_dims(peCos, axis = 0), [size, 1, 1])
    peSinY = tf.tile(tf.expand_dims(peSin, axis = 1), [1, size, 1])
    peCosY = tf.tile(tf.expand_dims(peCos, axis = 1), [1, size, 1])

    emb = tf.concat([peSinX, peCosX, peSinY, peCosY], axis = -1)
    return emb
# 2d positional embeddings of dimension 'dim' in a range of resolutions from 2x2 up to 'max_res x max_res'
#
# pos_type: supports several types of embedding schemes:
# - sinus: (see "Attention is all you need")
# - linear: where each position gets a value of [-1, 1] * trainable_vector, in each spatial
#   direction based on its location.
# - trainable: where an embedding of position [w,h] is [emb_w, emb_h] (independent parts in
#   each spatial direction)
# - trainable2d: where an embedding of position [w,h] is emb_{w,h} (a different embedding for
#   each position)
#
# dir_num: Each embedding consists of 'dir_num' parts, with each path measuring positional similarity
# along another direction, uniformly spanning the 2d space.
#
# shared: True for using same embeddings for all directions / parts
# init: uniform or normal distribution for trainable embeddings initialization
def get_positional_embeddings(max_res, dim, init = "uniform", shared = False):
    embs = []
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    for res in range(max_res + 1):
        with tf.variable_scope("pos_emb%d" % res):
            size = 2 ** res
            emb = get_sinusoidal_embeddings(size, dim)
            embs.append(emb)
    return embs
#---------------------------------------------------------------------------------------------

# Transformer (multi-head attention) function originated from the Google-BERT repository.
# https://github.com/google-research/bert/blob/master/modeling.py#L558
#
# We adopt their from/to notation:
# from_tensor: [batch_size, from_len, dim] a list of 'from_len' elements
# to_tensor: [batch_size, to_len, dim] a list of 'to_len' elements
#
# Each element in 'from_tensor' attends to elements from 'to_tensor',
# Then we compute a weighted sum over the 'to_tensor' elements, and use it to update
# the elements at 'from_tensor' (through additive/multiplicative integration).
#
# Overall it means that information flows in the direction to->from, or that the 'to'
# modulates the 'from'. For instance, if from=image, and to=latents, then the latents
# will control the image features. If from = to then this implements self-attention.
#
# We first project 'from_tensor' into a 'query', and 'to_tensor' into 'key' and 'value'.
# Then, the query and key tensors are dot-producted and softmaxed to obtain
# attention distribution over the to_tensor elements. The values are then
# interpolated (weighted-summed) using this distribution, to get 'context'.
# The context is used to modulate the bias/gain of the 'from_tensor' (depends on 'intervention').
# Notation: B - batch_size, F - from_len, T - to_len, N - num_heads, H - head_size
def transformer_layer(
        fmaps,                                  # The layer dimension
        from_tensor,        to_tensor,        # The from/to tensors [batch_size, t_len, dim] or [batch_size * t_len, dim]
        from_len = None,    to_len = None,    # The from/to tensors lengths (must be specified if from/to has 2 dims)
        from_pos = None,    to_pos = None,    # The positional encodings of the form/to tensors (optional)
        from_gate = False,  to_gate = False,  # Add sigmoid gate on from/to, so that info may not be sent/received
                                              # when gate is low (i.e. the attention probs may not sum to 1)
        # Additional options
        num_heads = 1,                        # Number of attention heads
        att_dp = 0.12,                        # Attention dropout rate
        att_mask = None,                      # Attention mask to block from/to elements [batch_size, from_len, to_len]
        integration = "mul",                  # Feature integration type: additive, multiplicative or both
        norm = "layer",                          # Feature normalization type (optional): instance, batch or layer

        # k-means options (optional, duplex)
        kmeans = False,                       # Track and update image-to-latents assignment centroids (in duplex)
        kmeans_iters = 1,                     # Number of K-means iterations per layer
        att_vars = {},                        # K-means variables carried over from layer to layer (only when --kmeans)
        iterative = False,                    # Whether to carry over attention assignments across transformer 
                                              # layers of different resolutions
        name = ""):                           # Layer variable_scope suffix

    # Validate input shapes and map them to 2d
    from_tensor, from_pos, from_shape, from_len, batch_size = process_input(from_tensor, from_pos, from_len, "from")
    to_tensor,   to_pos,   to_shape,   to_len,   _          = process_input(to_tensor, to_pos, to_len, "to")

    size_head = int(fmaps / num_heads)
    to_from = att_vars.get("centroid_assignments")

    with tf.variable_scope("AttLayer_{}".format(name)):
        # Compute queries, keys and values
        queries = apply_bias_act(dense_layer(from_tensor, fmaps, name = "query"), name = "query") # [B*F, N*H]
        keys    = apply_bias_act(dense_layer(to_tensor, fmaps, name = "key"), name = "key")       # [B*T, N*H]
        values  = apply_bias_act(dense_layer(to_tensor, fmaps, name = "value"), name = "value")   # [B*T, N*H]
        _queries = queries

        # Add positional encodings to queries and keys
        if from_pos is not None:
            queries += apply_bias_act(dense_layer(from_pos, fmaps, name = "from_pos"), name = "from_pos")
        if to_pos is not None:
            keys += apply_bias_act(dense_layer(to_pos, fmaps, name = "to_pos"), name = "to_pos")

        if kmeans:
            from_elements, to_centroids = compute_centroids(_queries, queries, to_from,
                to_len, from_len, batch_size, num_heads, size_head, parametric = not iterative)

        # Reshape queries, keys and values, and then compute att_scores
        values = transpose_for_scores(values, batch_size, num_heads, to_len, size_head)     # [B, N, T, H]
        queries = transpose_for_scores(queries, batch_size, num_heads, from_len, size_head) # [B, N, F, H]
        keys = transpose_for_scores(keys, batch_size, num_heads, to_len, size_head)         # [B, N, T, H]
        att_scores = tf.matmul(queries, keys, transpose_b = True)                           # [B, N, F, T]
        att_probs = None

        for i in range(kmeans_iters):
            with tf.variable_scope("iter_{}".format(i)):
                if kmeans:
                    if i > 0:
                        # Compute relative weights of different 'from' elements for each 'to' centroid
                        to_from = compute_assignments(att_probs)
                        # Given:
                        # 1. Centroid assignments of 'from' elements to 'to' centroid
                        # 2. 'from' elements (queries)
                        # Compute the 'to' respective centroids
                        to_centroids = tf.matmul(to_from, from_elements)

                    # Compute attention scores based on dot products between
                    # 'from' queries and the 'to' centroids.
                    w = tf.get_variable(name = "st_weights", shape = [num_heads, 1, get_shape(from_elements)[-1]],
                        initializer = tf.ones_initializer())
                    att_scores = tf.matmul(from_elements * w, to_centroids, transpose_b = True)

                # Scale attention scores given head size (see BERT)
                att_scores = tf.multiply(att_scores, 1.0 / math.sqrt(float(size_head)))
                # (optional, not used by default)
                # Mask attention logits using att_mask (to mask some components)
                if att_mask is not None:
                    att_scores = logits_mask(att_scores, tf.expand_dims(att_mask, axis = 1))
                # Turn attention logits to probabilities (softmax + dropout)
                att_probs = compute_probs(att_scores, att_dp)

        # Gate attention values for the from/to elements
        if to_gate:
            att_probs = gate_attention(att_probs, to_tensor, to_pos, batch_size, num_heads,
                from_len = 1, to_len = to_len, name = "e")
        if from_gate:
            att_probs = gate_attention(att_probs, from_tensor, from_pos, batch_size, num_heads,
                from_len = from_len, to_len = 1, name = "n", gate_bias = 1)

        # Compute relative weights of different 'from' elements for each 'to' centroid
        if kmeans:
            to_from = compute_assignments(att_probs)

        # Compute weighted-sum of the values using the attention distribution
        control = tf.matmul(att_probs, values) # [B, N, F, H]
        control = tf.transpose(control, [0, 2, 1, 3]) # [B, F, N, H]
        control = tf.reshape(control, [batch_size * from_len, fmaps]) # [B*F, N*H]
        # This newly computed information will control the bias/gain of the new from_tensor
        from_tensor = integrate(from_tensor, from_len, control, integration, norm)

    # Reshape from_tensor to its original shape (if 3 dimensions)
    if len(from_shape) > 2:
        from_tensor = tf.reshape(from_tensor, from_shape)

    return from_tensor, att_probs, {"centroid_assignments": to_from}

# (Not used be default)
# Merge multiple images together through a weighted sum to create one image (Used by k-GAN only)
# Also merge their the latent vectors that have used to create the images respectively
# Arguments:
# - x: the image features, [batch_size * k, C, H, W]
# - k: the number of images to merge together
# - type: the type of the merge (sum, softmax, max, leaves)
# - same: whether to merge all images equally along all WH positions
# Returns the merged images [batch_size, C, H, W] and latents [batch_size, k, layers_num, dim]
def merge_images(x, dlatents, k, type, same = False):
    # Reshape function to have the k copies to be merged 
    def k_reshape(t): return tf.reshape(t, [-1, k] + get_shape(t)[1:])

    # Compute scores w to be used in a following weighted sum of the images x
    scores = k_reshape(conv2d_layer(x, dim = 1, kernel = 1)) # [B, k, 1, H, W]
    if same:
        scores = tf.reduce_sum(scores, axis = [-2, -1], keepdims = True) # [B, k, 1, 1, 1]

    # Compute soft probabilities for weighted sum
    if type == "softmax":
        scores = tf.nn.softmax(scores, axis = 1) # [B, k, 1, H, W]
    # Take only image of highest score
    elif type == "max":
        scores = tf.one_hot(tf.math.argmax(scores, axis = 1), k, axis = 1) # [B, k, 1, H, W]
    # Merge the images recursively using a falling-leaves scheme (see k-GAN paper for details)
    elif type == "leaves":
        score_list = tf.split(scores, k, axis = 1) # [[B, 1, H, W],...]
        alphas = tf.ones_like(score_list[0]) # [B, 1, H, W]
        for d, weight in enumerate(score_list[1:]): # Process the k images iteratively
            max_weight = tf.math.reduce_max(scores[:,:d + 1], ) # Compute image currently most "in-front"
            new_alphas = tf.sigmoid(weight - max_weight) # Compute alpha values based on "distance" to the new image
            alphas = tf.concat([alphas * (1 - new_alphas), new_alphas], axis = 1) # Compute recursive alpha values
        scores = alphas
    else: # sum
        scores = tf.ones_like(scores) # [B, k, 1, H, W]

    # Compute a weighted sum of the images
    x = tf.reduce_sum(k_reshape(x) * scores, axis = 1) # [B, C, H, W]

    if dlatents is not None:
        scores = tf.reduce_mean(tf.reduce_mean(scores, axis = 2), axis = [-2, -1], keepdims = True) # [B, k, 1, 1]
        dlatents = tf.tile(tf.reduce_sum(dlatents * scores, axis = 1, keepdims = True), [1, k, 1, 1]) # [B, k, L, D]

    return x, dlatents
#----------------------------------------------------------------------------
# Generator GANFormer
def G_GANformer(
    latents_in,                               # First input: Latent vectors (z) [batch_size, latent_size]
    labels_in,                                # Second input (optional): Conditioning labels [batch_size, label_size]
    # General arguments
    is_training             = False,          # Network is under training? Enables and disables specific features
    is_validation           = False,          # Network is under validation? Chooses which value to use for truncation_psi
    return_dlatents         = False,          # Return dlatents in addition to the images?
    take_dlatents           = False,          # Use latents_in as dlatents (skip mapping network)
    is_template_graph       = False,          # True = template graph constructed by the Network class, False = actual evaluation
    components              = dnnlib.EasyDict(),     # Container for sub-networks. Retained between calls
    mapping_func            = "G_mapping",    # Function name of the mapping network
    synthesis_func          = "G_synthesis_GANFormer",  # Function name of the synthesis network
    # Truncation cutoff
    truncation_psi          = 0.7,            # Style strength multiplier for the truncation trick. None = disable
    truncation_cutoff       = None,           # Number of layers for which to apply the truncation trick. None = disable
    truncation_psi_val      = None,           # Value for truncation_psi to use during validation
    truncation_cutoff_val   = None,           # Value for truncation_cutoff to use during validation
    # W latent space mean tracking
    dlatent_avg_beta        = 0.995,          # Decay for tracking the moving average of W during training. None = disable
    # Mixing
    style_mixing            = 0.9,            # Probability of mixing styles during training. None = disable
    component_mixing        = 0.0,            # Probability of mixing components during training. None = disable
    component_dropout       = 0.0,            # Dropout over the k latent components 0 = disable
    **kwargs):                                # Arguments for sub-networks (mapping and synthesis)

    # Validate arguments
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    latents_in = tf.cast(latents_in, tf.float32)
    labels_in = tf.cast(labels_in, tf.float32)

    # Set options for training/validation
    # Set truncation_psi values if validations
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    # Turn off truncation cutoff for training
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    # Turn off update of w latent mean when not training
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    # Turn off style and component mixing when not training
    if not is_training or (style_mixing is not None and not tflib.is_tf_expression(style_mixing) and style_mixing <= 0):
        style_mixing = None
    if not is_training or (component_mixing is not None and not tflib.is_tf_expression(component_mixing) and component_mixing <= 0):
        component_mixing = None
    # Turn off dropout when not training
    if not is_training:
        kwargs["attention_dropout"] = 0.0

    # Useful variables
    k = kwargs["components_num"]
    latent_size = kwargs["latent_size"]
    dlatent_size = kwargs["latent_size"]

    latents_num = k + 1 
    # Setup sub-networks
    # Set synthesis network
    if "synthesis" not in components:
        components.synthesis = tflib.Network("G_synthesis", func_name = globals()[synthesis_func], **kwargs)
    num_layers = components.synthesis.input_shape[2]
    # Set mapping network
    if "mapping" not in components:
        components.mapping = tflib.Network("G_mapping", func_name = globals()[mapping_func],
            dlatent_broadcast = num_layers, **kwargs)

    # If latents_in used for dlatents (skipping mapping network),
    # then need latent per each synthesis network layer, see StyleGAN for details.
    if take_dlatents:
        latents_in.set_shape([None, latents_num, num_layers, latent_size])
    else:
        latents_in.set_shape([None, latents_num, latent_size])
    batch_size = get_shape(latents_in)[0]

    # Initialize trainable positional embeddings for k latent components
    latent_pos = get_embeddings(k, dlatent_size, name = "ltnt_emb")
    # Initialize component dropout mask (disabled by default)
    component_mask = random_dp_binary([batch_size, k], component_dropout)
    component_mask = tf.expand_dims(component_mask, axis = 1)

    # Setup variables
    dlatent_avg = tf.get_variable("dlatent_avg", shape = [dlatent_size], 
        initializer = tf.initializers.zeros(), trainable = False)

    if take_dlatents:
        dlatents = latents_in
    else:
        # Evaluate mapping network
        dlatents = components.mapping.get_output_for(latents_in, labels_in, latent_pos, 
            component_mask, is_training = is_training, **kwargs)
        dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W latent space
    if dlatent_avg_beta is not None:
        with tf.variable_scope("DlatentAvg"):
            batch_avg = tf.reduce_mean(dlatents[:, :, 0], axis = [0, 1])
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Mixing (see StyleGAN): mixes together some of the W latents mapped from one set of
    # source latents A, and another set of source latents B. Mixing can be between latents
    # that correspond either to different layers or to the different k components used in
    # the GANformer.
    # Used only during training, and by default only layers are mixes as in StyleGAN.
    def mixing(latents_in, dlatents, prob, num, idx):
        if prob is None or prob == 0:
            return dlatents

        with tf.variable_scope("StyleMix"):
            latents2 = tf.random_normal(get_shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, latent_pos, 
                component_mask, is_training = is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < prob,
                lambda: tf.random_uniform([], 1, num, dtype = tf.int32),
                lambda: num)
            dlatents = tf.where(tf.broadcast_to(idx < mixing_cutoff, get_shape(dlatents)), dlatents, dlatents2)
        return dlatents

    # Perform style mixing regularization
    layer_idx = np.arange(num_layers)[np.newaxis, np.newaxis, :, np.newaxis]
    dlatents = mixing(latents_in, dlatents, style_mixing, num_layers, layer_idx)
    # Perform component mixing regularization
    ltnt_idx = np.arange(latents_num)[np.newaxis, :, np.newaxis, np.newaxis]
    dlatents = mixing(latents_in, dlatents, component_mixing, k, ltnt_idx)

    # Apply truncation (not in training or evaluation, only when sampling images, see StyleGAN for details)
    if truncation_psi is not None:
        with tf.variable_scope("Truncation"):
            layer_idx = np.arange(num_layers)[np.newaxis, np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype = np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = tflib.lerp(dlatent_avg, dlatents, layer_psi)

    # Evaluate synthesis network
    imgs_out, maps_out = components.synthesis.get_output_for(dlatents, latent_pos, component_mask,
        is_training = is_training, force_clean_graph = is_template_graph, **kwargs)

    # Return requested outputs
    imgs_out = tf.identity(imgs_out, name = "images_out") # [batch_size, num_channels, H, W]
    maps_out = tf.identity(maps_out, name = "maps_out") # [batch_size, k (components_num), layers_num, heads_num, H, W]
    ret = (imgs_out, maps_out)

    if return_dlatents:
        dlatents = tf.identity(dlatents, name = "dlatents_out")
        ret += (dlatents,) # [batch_size, dlatent_broadcast, k (components_num), dlatent_dim]

    return ret


########################################### Mapping network ###########################################

# Mostly similar to the StylGAN version, with new options for the GANformer such as:
# self-attention among latents, shared mapping of components, and support for mapping_dim != output_dim
def G_mapping(
    # Tensor inputs
    latents_in,                             # First input: Latent vectors (z) [batch_size, latent_size]
    labels_in,                              # Second input (optional): Conditioning labels [batch_size, label_size]
    latent_pos,                             # Third input: Positional embeddings for latents
    component_mask,                         # Fourth input: Component dropout mask (not used by default)
    # Dimensions
    components_num          = 16,            # Number of latent (z) vector components z_1,...,z_k
    latent_size             = 512,          # Latent (z) dimensionality per component
    dlatent_size            = 512,          # Disentangled latent (w) dimensionality
    label_size              = 0,            # Label dimensionality, 0 if no labels
    # General settings
    dlatent_broadcast       = None,         # Output disentangled latent (w) as [batch_size, dlatent_size]
                                            # or [batch_size, dlatent_broadcast, dlatent_size]
    normalize_latents       = True,         # Normalize latent vectors (z) before feeding them to the mapping layers?
    # Mapping network
    mapping_layersnum       = 8,            # Number of mapping layers
    mapping_dim             = None,         # Number of activations in the mapping layers
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers
    mapping_nonlinearity    = "lrelu",      # Activation function: relu, lrelu, etc.
    mapping_ltnt2ltnt       = False,        # Add self-attention over latents in the mapping network
    mapping_resnet          = True,        # Use resnet connections in mapping network
    mapping_shared_dim      = 0,            # Perform a shared mapping with that dimension to all latents concatenated together
    # Attention options
    num_heads               = 1,            # Number of attention heads
    use_pos                 = True,        # Use positional encoding for latents
    ltnt_gate               = False,        # Gate attention scores so that info may not be sent/received when value is low
    attention_dropout       = 0.12,         # Attention dropout rate
    **_kwargs):                             # Ignore unrecognized keyword args

    # Short names
    act = mapping_nonlinearity
    k = components_num
    latents_num = k + 1

    net_dim = mapping_dim
    layersnum = mapping_layersnum
    lrmul = mapping_lrmul
    ltnt2ltnt = mapping_ltnt2ltnt
    resnet = mapping_resnet
    shared_dim = mapping_shared_dim

    # Inputs
    latents_in.set_shape([None, latents_num, latent_size])
    labels_in.set_shape([None, label_size])
    latent_pos.set_shape([k, dlatent_size])
    component_mask.set_shape([None, 1, k])

    batch_size = get_shape(latents_in)[0]

    if not use_pos:
        latent_pos = None

    x = latents_in

    # Set internal and output dimensions
    if net_dim is None:
        net_dim = dlatent_size
        output_dim = None # Since mapping dimension is already dlatent_size
    else:
        output_dim = dlatent_size
        # Map inputs to the mapping dimension
        x = to_2d(x, "last")
        x = apply_bias_act(dense_layer(x, net_dim, name = "map_start"), name = "map_start")
        x = tf.reshape(x, [batch_size, latents_num, net_dim])
        if latent_pos is not None:
            latent_pos = apply_bias_act(dense_layer(latent_pos, net_dim, name = "map_pos"), name = "map_pos")


    # Split latents to region-based and global
    x, g = tf.split(x, [k, 1], axis = 1)
    g = tf.squeeze(g, axis = 1)

    # Normalize latents
    if normalize_latents:
        with tf.variable_scope("Normalize"):
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = -1, keepdims = True) + 1e-8)
    mlp_kwargs = {}
    if ltnt2ltnt:
        mlp_kwargs.update({        
        "transformer": ltnt2ltnt,
        "num_heads": num_heads, 
        "att_dp": attention_dropout,
        "from_gate": ltnt_gate, 
        "to_gate": ltnt_gate,
        "from_pos": latent_pos, 
        "to_pos": latent_pos,
        "from_len": k,          
        "to_len": k
        })
    # Mapping layers
    if k == 0:
        x = tf.zeros([batch_size, 0, net_dim])
    else:
        # If shared mapping is set, then:
        if shared_dim > 0:
            # Concatenate all latents together (taken care by dense_layer function)
            x = apply_bias_act(dense_layer(x, shared_dim, name = "inmap"), name = "inmap")
            # Map the latents to the w space
            x = mlp(x, resnet, layersnum, shared_dim, act, lrmul, **mlp_kwargs)
            # Split the latents back to the k components
            x = tf.reshape(apply_bias_act(dense_layer(x, k * net_dim, name = "outmap"), name = "outmap"),
                [batch_size, k, net_dim])
        else:
            x = mlp(x, resnet, layersnum, net_dim, act, lrmul, pooling = "batch",
                att_mask = component_mask, **mlp_kwargs)

    with tf.variable_scope("global"):
        # Map global latent to the w space
        g = mlp(g, resnet, layersnum, net_dim, act, lrmul)
    # Concatenate back region-based and global latents
    x = tf.concat([x, tf.expand_dims(g, axis = 1)], axis = 1)

    # Map latents to output dimension
    if output_dim is not None:
        x = tf.reshape(apply_bias_act(dense_layer(to_2d(x, "last"), output_dim, name = "map_end"), name = "map_end"),
            [batch_size, latents_num, output_dim])

    # Broadcast latents to all layers (so to control each of the resolution layers of the Synthesis network)
    if dlatent_broadcast is not None:
        with tf.variable_scope("Broadcast"):
            x = tf.tile(x[:, :, np.newaxis], [1, 1, dlatent_broadcast, 1])

    # Output
    x = tf.identity(x, name = "dlatents_out")
    return x # [batch_size, dlatent_broadcast, k (components_num), dlatent_dim]


########################################## Synthesis network ##########################################

# Main differences from the StyleGAN version include the incorporation of transformer layers.
def G_synthesis_GANFormer(
    # Tensor inputs
    dlatents_in,                        # First input: Disentangled latents (W) [batch_size, num_layers, dlatent_size]
    latent_pos,                         # Second input: Positional embeddings for latents
    component_mask,                     # Third input: Component dropout mask (not used by default)
    # Dimensions and resolution
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality
    pos_dim             = None,         # Positional embeddings dimension
    num_channels        = 3,            # Number of output color channels
    resolution          = 1024,         # Output resolution
    fmap_base           = 16 << 10,     # Overall multiplier for the network dimension
    fmap_decay          = 1.0,          # log2 network dimension reduction when doubling the resolution
    fmap_min            = 1,            # Minimum network dimension in any layer
    fmap_max            = 512,          # Maximum network dimension in any layer
    # General settings (StyleGAN/GANformer related)
    architecture        = "resnet",       # Architecture: orig, skip, resnet
    nonlinearity        = "lrelu",      # Activation function: relu, lrelu", etc.
    resample_kernel     = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations, None = no filtering
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    latent_stem         = False,        # If True, map latents to initial 4x4 image grid. Otherwise, use a trainable constant grid
    local_noise         = True,         # Add local stochastic noise during image synthesis
                                        # Otherwise, read noise inputs from variables
    # GANformer settings
    components_num      = 16,            # Number of Latent (z) vector components z_1,...,z_k
    num_heads           = 1,            # Number of attention heads
    attention_dropout   = 0.12,         # Attention dropout rate
    ltnt_gate           = False,        # Gate attention from latents, such that components may not send information
                                        # when gate value is low
    img_gate            = False,        # Gate attention for images, such that some image positions may not get updated
                                        # or receive information when gate value is low
    integration         = "mul",        # Feature integration type: additive, multiplicative or both
    norm                = "layer",         # Feature normalization type (optional): instance, batch or layer
    # Extra attention options, related to key-value duplex attention
    kmeans              = False,        # Track and update image-to-latents assignment centroids, used in the duplex attention
    kmeans_iters        = 1,            # Number of K-means iterations per transformer layer
    iterative           = False,        # Carry over attention assignments across transformer layers of different resolutions
                                        # If True, centroids are carried from layer to layer
    # Attention directions and layers:
    # image->latents, latents->image, image->image (SAGAN), resolution to be applied at
    start_res           = 0,            # Transformer minimum resolution layer to be used at
    end_res             = 20,           # Transformer maximum resolution layer to be used at
    img2ltnt            = False,        # Add image to latents attention (bottom-up)
    # Positional encoding
    use_pos             = True,        # Use positional encoding for latents
    pos_init            = "uniform",    # Positional encoding initialization distribution: normal or uniform
    **_kwargs):                         # Ignore unrecognized keyword args

    # Set variables
    k = components_num
    act = nonlinearity
    latents_num = k + 1
    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 * 2 - 1
    if pos_dim is None:
        pos_dim = dlatent_size

    assert resolution == 2**resolution_log2 and resolution >= 4
    assert architecture in ["orig", "skip", "resnet"]

    # Network dimension, is set to fmap_base and then reduces with increasing stage
    # according to fmap_decay, in the range of [fmap_min, fmap_max] (see StyleGAN)
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    # Get global latent (the last one)
    def get_global(dlatents, res):
        return dlatents[:, -1]


    # Inputs
    dlatents_in.set_shape([None, latents_num, num_layers, dlatent_size])
    component_mask.set_shape([None, 1, k])
    latent_pos.set_shape([k, dlatent_size])
    # Disable latent_pos if we don't want to use it
    if not use_pos:
        latent_pos = None

    batch_size = get_shape(dlatents_in)[0]

    # Positional encodings (spatial)
    # Image positional encodings, dictionary at different resolutions (used in GANformer)
    grid_poses = get_positional_embeddings(resolution_log2, pos_dim, pos_init)

    # Local noise inputs to add to features during image synthesis (if --local-noise)
    noise_layers = []
    for layer_idx in range(num_layers - 1):
        # Infer layer resolution from its index
        res = (layer_idx + 5) // 2
        batch_multiplier = 1
        noise_shape = [batch_multiplier, 1, 2**res, 2**res]



        # Create new noise variable
        noise_layers.append(tf.get_variable("noise%d" % layer_idx, shape = noise_shape,
            initializer = tf.initializers.random_normal(), trainable = False))


    # Single generator layer with transformer (optionally)
    def layer(x, dlatents, layer_idx, dim, kernel, att_vars, up = False):
        att_map = None
        # Image features resolution of current layer
        res = (layer_idx + 5) // 2
        # Global latent for features global modulation (as in StyleGAN)
        dlatent_global = get_global(dlatents_in, res)[:, layer_idx + 1]
        # If bottom-up connections exist, use the (iteratively updated) latents
        # Otherwise, use the input latents corresponding to the current layer
        new_dlatents = None
        if dlatents is None:
            dlatents = dlatents_in[:, :-1, layer_idx + 1]
        _fused_modconv = fused_modconv

        # Modulate image features and perform convolution
        x = modulated_conv2d_layer(x, dlatent_global, dim, kernel, up = up,
            resample_kernel = resample_kernel, fused_modconv = _fused_modconv)
        shape = get_shape(x)

        # Transformer layer
        if res >= start_res and res < end_res:
            # Reshape image features for transformer layer [batch_size (B), img_feats (WH), feat_dim (C)]
            x = tf.transpose(tf.reshape(x, [shape[0], shape[1], shape[2] * shape[3]]), [0, 2, 1])

            ################################## Top-down latents->image ##################################
            # Main Transformer layer: arguments and function call
            kwargs = {
                "num_heads": num_heads,         # Number of attention heads
                "integration": integration,     # Integration mode: additive, multiplicative, or both
                "norm": norm,                   # Feature normalization type: batch, instance of layer
                "att_mask": component_mask,     # Attention mask (to disable particular latents)
                "att_dp": attention_dropout,    # Attention dropout rate
                "from_gate": img_gate,          # Gate attention flow from images (see G_synthesis signature for details)
                "to_gate": ltnt_gate,           # Gate attention flow to latents (see G_synthesis signature for details)
                "from_pos": grid_poses[res],    # Positional encodings for the image
                "to_pos": latent_pos,           # Positional encodings for the latents
                "kmeans": kmeans,               # Track and update image-to-latents assignment centroids (in duplex)
                "kmeans_iters": kmeans_iters,   # Number of K-means iterations per layer
                "att_vars": att_vars,           # K-means variables carried over from layer to layer (only when --kmeans)
                "iterative": iterative          # Whether to carry over attention assignments across transformer 
                                                # layers of different resolutions
            }
            # Perform attention from image to latents, meaning that information will flow
            # latents -> image (from/to naming matches with Google BERT repository)
            x, att_map, att_vars = transformer_layer(from_tensor = x, to_tensor = dlatents, fmaps = dim,
                name = "l2n", **kwargs)

            ################################ Optional transformer layers ################################
            # image->image, image->latents, latents->latents
            kwargs = {
                "num_heads": num_heads,   "att_dp": attention_dropout,
                "from_gate": ltnt_gate,   "from_pos": latent_pos
            }

            # Bottom-Up: Image->Latent attention
            if img2ltnt:
                new_dlatents = nnlayer(dlatents, dlatent_size, y = x, act = "lrelu", pool = True, 
                    name = "n2l", to_pos = grid_poses[res], to_gate = img_gate, **kwargs)
            # Reshape image features back to standard [NCHW]
            x = tf.reshape(tf.transpose(x, [0, 2, 1]), shape)

        # Add local stochastic noise to image features
        if local_noise:
            shape = get_shape(x) # NCHW
            shape[1] = 1
            noise = tf.random_normal(shape)
            strength = tf.get_variable("noise_strength", shape = [], initializer = tf.initializers.zeros())
            x += strength * noise
            # Biases and nonlinearity
            x = apply_bias_act(x, act = act)

        return x, new_dlatents, att_map, att_vars

    # The building block for the generator (two layers with upsampling and a resnet connection)
    def block(x, res, dlatents, dim, att_vars, up = True): # res = 3..resolution_log2
        t = x
        with tf.variable_scope("Conv0_up"):
            x, dlatents, att_map1, att_vars = layer(x, dlatents, layer_idx = res*2-5,
                dim = dim, kernel = 3, up = up, att_vars = att_vars)

        with tf.variable_scope("Conv1"):
            x, dlatents, att_map2, att_vars = layer(x, dlatents, layer_idx = res*2-4,
                dim = dim, kernel = 3, att_vars = att_vars)

        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, fmaps = dim, kernel = 1, up = up, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))

        att_maps = [att_map1, att_map2]
        return x, dlatents, att_maps, att_vars

    def upsample(y):
        with tf.variable_scope("Upsample"):
            return upsample_2d(y, k = resample_kernel)

    # Convert image features to output image (e.g. RGB)
    # - For GANformer: optionally perform transformer on highest resolution (if end_res >= resolution_log2)
    def torgb(t, y, res, dlatents): # res = 2..resolution_log2
        with tf.variable_scope("ToRGB"):
            if res == resolution_log2:
                # If end_res >= resolution_log2, need to perform transformer on highest resolution too
                if res <= end_res:
                    with tf.variable_scope("extraLayer"):
                        t = modulated_conv2d_layer(t, dlatents[:, res*2-3], fmaps = nf(res-1),
                            kernel = 3, fused_modconv = fused_modconv)

            # Convert image features to output image (with num_channels, e.g. RGB)
            t = modulated_conv2d_layer(t, dlatents[:, res*2-2], fmaps = num_channels,
                kernel = 1, demodulate = False, fused_modconv = fused_modconv)
            t = apply_bias_act(t)

            # Optional skip connections (see StyleGAN)
            if y is not None:
                t += y

            return t

    # Early layers
    imgs_out, dlatents, att_maps = None, None, []
    att_vars = {"centroid_assignments": None}

    # x is img_feats: the evolving image features grid, starting from 4x4, and
    # iteratively processed and upsampled up to the desirable resolution.

    # Stem: initial 4x4 image grid -- either constant (StyleGAN) or mapped from the latents (standard GAN)
    with tf.variable_scope("4x4"):
        # Default option, trainable constant 4x4 image grid stem
        if not latent_stem:
            with tf.variable_scope("Const"):
                stem_size = 1
                x = tf.get_variable("const", shape = [stem_size, nf(1), 4, 4],
                    initializer = tf.initializers.random_normal())
                x = tf.tile(x, [batch_size, 1, 1, 1])
        # First modulation layer over 4x4
        with tf.variable_scope("Conv"):
            x, dlatents, att_map, att_vars = layer(x, dlatents, layer_idx = 0, dim = nf(1),
                kernel = 3, att_vars = att_vars)
            att_maps.append(att_map)
        if architecture == "skip":
            imgs_out = torgb(x, imgs_out, 2, get_global(dlatents_in, res))

    # Main layers
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope("%dx%d" % (2**res, 2**res)):
            # Generator block: transformer, convolution and upsampling
            x, dlatents, _att_maps, att_vars = block(x, res, dlatents, dim = nf(res-1), att_vars = att_vars)
            att_maps += _att_maps
            # Optional skip/resnet connections (see StyleGAN)
            if architecture == "skip" or res == resolution_log2:
                if architecture == "skip":
                    imgs_out = upsample(imgs_out)
                imgs_out = torgb(x, imgs_out, res, get_global(dlatents_in, res))

    # Convert the list of all attention maps from all layers into one tensor
    def list2tensor(att_list):
        att_list = [att_map for att_map in att_list if att_map is not None]
        if len(att_list) == 0:
            return None

        maps_out = []
        for att_map in att_list:
            # Reshape attention map into spatial
            s = int(math.sqrt(get_shape(att_map)[2]))
            att_map = tf.transpose(tf.reshape(att_map, [-1, s, s, k]), [0, 3, 1, 2]) # [NCHW]
            # Upsample attention map to final image resolution
            # (since attention map of early generator layers have lower resolutions)
            if s < resolution:
                att_map = upsample_2d(att_map, factor = int(resolution / s))
            att_map = tf.reshape(att_map, [-1, num_heads, k, resolution, resolution]) # [NhkHW]
            maps_out.append(att_map)

        maps_out = tf.transpose(tf.stack(maps_out, axis = 1), [0, 3, 1, 2, 4, 5]) # [NklhHW]
        return maps_out

    # Output
    maps_out = list2tensor(att_maps)

    # imgs_out [batch_size, num_channels, H, W]
    # maps_out [batch_size, k (components_num), layers_num, heads_num, H, W]
    return imgs_out, maps_out

# Discriminator network with Attention as in the paper.
def D_GANformer(
    images_in,                          # First input: Images [batch_size, channel, height, width]
    labels_in,                          # Second input: Labels [batch_size, label_size]
    # Dimensions and resolution
    latent_size         = 512,          # Aggregator variables dimension (only for using transformer in discriminator)
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels
    pos_dim             = None,         # Positional embeddings dimension
    num_channels        = 3,            # Number of input color channels
    resolution          = 1024,         # Input resolution
    fmap_base           = 16 << 10,     # Overall multiplier for the network dimension
    fmap_decay          = 1.0,          # log2 network dimension reduction when doubling the resolution
    fmap_min            = 1,            # Minimum network dimension in any layer
    fmap_max            = 512,          # Maximum network dimension in any layer
    # General settings
    architecture        = "resnet",     # Architecture: orig, skip, resnet
    nonlinearity        = "lrelu",      # Activation function: relu, lrelu, etc
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer
    resample_kernel     = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations, None = no filtering
    # GANformer settings
    components_num      = 16,            # Number of aggregator variables
    num_heads           = 1,            # Number of attention heads
    attention_dropout   = 0.12,         # Attention dropout rate
    ltnt_gate           = False,        # Gate attention from latents, such that components may not send information
                                        # when gate value is low
    img_gate            = False,        # Gate attention for images, such that some image positions may not get updated
                                        # or receive information when gate value is low
    # Attention directions and layers
    # image->latents, latents->image, image->image (SAGAN), resolution to be applied at
    start_res           = 0,            # Transformer minimum resolution layer to be used at
    end_res             = 20,           # Transformer maximum resolution layer to be used at
    ltnt2img            = False,        # Add aggregators to image attention (top-down)
    # Positional encoding
    use_pos             = True,        # Use positional encoding for latents
    pos_init            = "uniform",    # Positional encoding initialization distribution: normal or uniform
    **_kwargs):                         # Ignore unrecognized keyword args

    # Set variables
    act = nonlinearity
    resolution_log2 = int(np.log2(resolution))
    if pos_dim is None:
        pos_dim = latent_size

    assert architecture in ["orig", "skip", "resnet"]
    assert resolution == 2**resolution_log2 and resolution >= 4

    # Network dimension, is set to fmap_base and then reduces with increasing stage
    # according to fmap_decay, in the range of [fmap_min, fmap_max] (see StyleGAN)
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    # Inputs
    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])

    images_in = tf.cast(images_in, tf.float32)
    labels_in = tf.cast(labels_in, tf.float32)

    batch_size = get_shape(images_in)[0]

    # Positional encodings
    latent_pos = None
    if use_pos:
        latent_pos = get_embeddings(components_num, latent_size, name = "ltnt_emb")

    grid_poses = get_positional_embeddings(resolution_log2, pos_dim, pos_init)

    # Convert input image (e.g. RGB) to image features
    def fromrgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope("FromRGB"):
            t = apply_bias_act(conv2d_layer(y, fmaps = nf(res-1), kernel = 1), act = act)
            # Optional skip connections (see StyleGAN)
            if x is not None:
                t += x
            return t

    # The building block for the discriminator
    def block(x, res, aggregators): # res = 2..resolution_log2
        shape = get_shape(x)
        t = x

        ################################## Transformer layers ##################################
        if res >= start_res and res < end_res:
            # Reshape image features for transformer layer [batch_size (B), img_feats (WH), feat_dim (C)]
            x = tf.transpose(tf.reshape(x, [batch_size, shape[1], shape[2] * shape[3]]), [0, 2, 1])

            # Main image->Latents (bottom-up) transformer layer
            kwargs = {
                "num_heads": num_heads,  "att_dp": attention_dropout,
                "from_gate": ltnt_gate,  "to_gate": img_gate,
                "from_pos": latent_pos,  "to_pos": grid_poses[res]
            }
            aggregators = transformer_layer(from_tensor = aggregators, to_tensor = x, fmaps = latent_size, name = "n2l", **kwargs)[0]

            kwargs.update({"to_gate": ltnt_gate, "to_pos": latent_pos})

            # Optional transformer layers
            # Latents->image (top-down)
            if ltnt2img:
                kwargs.update({"from_gate": img_gate, "from_pos": grid_poses[res]})
                x = transformer_layer(from_tensor = x, to_tensor = aggregators, fmaps = get_shape(x)[-1], name = "l2n", **kwargs)[0]

            # Reshape image features back to standard [NCHW]
            x = tf.reshape(tf.transpose(x, [0, 2, 1]), shape)

        ksize = 3 # 3x3 convolution
        ############################# Convolution layers (standard) ############################
        # Two convolution layers with downsmapling and a resnet connection
        with tf.variable_scope("Conv0"):
            x = apply_bias_act(conv2d_layer(x, fmaps = nf(res-1), kernel = ksize), act = act)

        with tf.variable_scope("Conv1_down"):
            x = apply_bias_act(conv2d_layer(x, fmaps = nf(res-2), kernel = ksize, down = True,
                resample_kernel = resample_kernel), act = act)

        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, fmaps = nf(res-2), kernel = 1, down = True, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))

        return x, aggregators

    def downsample(y):
        with tf.variable_scope("Downsample"):
            return downsample_2d(y, k = resample_kernel)

    # x is img_feats: the evolving image features grid, starting from the input resolution, and iteratively
    # processed and downsampled until making final binary prediction about the image source (real / fake).
    x = None

    # Aggregator variables set be to used with attention
    aggregators = tf.get_variable(name = "aggregators", shape = [components_num, latent_size],
        initializer = tf.random_uniform_initializer())
    aggregators = tf.tile(tf.expand_dims(aggregators, axis = 0), [batch_size, 1, 1]) # [batch_size, k, dim]

    # Main layers
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope("%dx%d" % (2**res, 2**res)):
            # Optional skip/resnet connections (see StyleGAN)
            if architecture == "skip" or res == resolution_log2:
                x = fromrgb(x, images_in, res)
            # Process features through the discriminator block
            x, aggregators = block(x, res, aggregators)
            # Downsampling for skip connection
            if architecture == "skip":
                images_in = downsample(images_in)

    # Final layers
    with tf.variable_scope("4x4"):
        if architecture == "skip":
            x = fromrgb(x, images_in, 2)
        # Minibatch standard deviation layer (see StyleGAN)
        if mbstd_group_size > 1:
            with tf.variable_scope("MinibatchStddev"):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope("Conv"):
            x = apply_bias_act(conv2d_layer(x, fmaps = nf(1), kernel = 3), act = act)
        # Turn final 4x4 image grid to a vector
        with tf.variable_scope("Dense0"):
            x = apply_bias_act(dense_layer(x, fmaps = nf(0)), act = act) # [batch_size, nf(0)]

    with tf.variable_scope("ComponentScores"):
        if mbstd_group_size > 1:
            with tf.variable_scope("MinibatchStddev"):
                aggregators = tf.transpose(aggregators, [0, 2, 1])
                aggregators = minibatch_stddev_layer(aggregators, mbstd_group_size, mbstd_num_features, have_last = False)
                aggregators = tf.transpose(aggregators, [0, 2, 1])

        shape = get_shape(aggregators) # [batch_size, k, dim]

        with tf.variable_scope("Dense"):
            aggregators = apply_bias_act(dense_layer(to_2d(aggregators, "last"), latent_size), act = act) # [batch_size * k, dim]
        with tf.variable_scope("Output"):
            o = apply_bias_act(dense_layer(aggregators, 1)) # [batch_size * k, 1]

        o = tf.reshape(o, shape[:-1]) # [batch_size, k]
        x = tf.concat([x, o], axis = -1) # [batch_size, nf(0) + k]

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope("Output"):
        x = apply_bias_act(dense_layer(x, fmaps = max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis = 1, keepdims = True) # [batch_size, 1]

    # Output
    scores_out = tf.identity(x, name = "scores_out")
    return scores_out # [batch_size, 1]


#
############################################ Discriminator ############################################
# -----------------------------------------------------------------------------------------------------

# Discriminator network, with convolution, downsampling, and optional transformer layers
def D_Stylegan(
    images_in,                          # First input: Images [batch_size, channel, height, width]
    labels_in,                          # Second input: Labels [batch_size, label_size]
    # Dimensions and resolution
    latent_size         = 512,          # Aggregator variables dimension (only for using transformer in discriminator)
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels
    pos_dim             = None,         # Positional embeddings dimension
    num_channels        = 3,            # Number of input color channels
    resolution          = 1024,         # Input resolution
    fmap_base           = 16 << 10,     # Overall multiplier for the network dimension
    fmap_decay          = 1.0,          # log2 network dimension reduction when doubling the resolution
    fmap_min            = 1,            # Minimum network dimension in any layer
    fmap_max            = 512,          # Maximum network dimension in any layer
    # General settings
    architecture        = "resnet",     # Architecture: orig, skip, resnet
    nonlinearity        = "lrelu",      # Activation function: relu, lrelu, etc
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer
    resample_kernel     = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations, None = no filtering
    **_kwargs):                         # Ignore unrecognized keyword args

    # Set variables
    act = nonlinearity
    resolution_log2 = int(np.log2(resolution))
    if pos_dim is None:
        pos_dim = latent_size

    assert architecture in ["orig", "skip", "resnet"]
    assert resolution == 2**resolution_log2 and resolution >= 4

    # Network dimension, is set to fmap_base and then reduces with increasing stage
    # according to fmap_decay, in the range of [fmap_min, fmap_max] (see StyleGAN)
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    # Inputs
    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])

    images_in = tf.cast(images_in, tf.float32)
    labels_in = tf.cast(labels_in, tf.float32)

    # Convert input image (e.g. RGB) to image features
    def fromrgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope("FromRGB"):
            t = apply_bias_act(conv2d_layer(y, fmaps = nf(res-1), kernel = 1), act = act)
            # Optional skip connections (see StyleGAN)
            if x is not None:
                t += x
            return t

    # The building block for the discriminator
    def block(x, res): # res = 2..resolution_log2
        shape = get_shape(x)
        t = x


        ksize = 3 # 3x3 convolution
        ############################# Convolution layers (standard) ############################
        # Two convolution layers with downsmapling and a resnet connection
        with tf.variable_scope("Conv0"):
            x = apply_bias_act(conv2d_layer(x, fmaps = nf(res-1), kernel = ksize), act = act)

        with tf.variable_scope("Conv1_down"):
            x = apply_bias_act(conv2d_layer(x, fmaps = nf(res-2), kernel = ksize, down = True,
                resample_kernel = resample_kernel), act = act)

        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, fmaps = nf(res-2), kernel = 1, down = True, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))

        return x

    def downsample(y):
        with tf.variable_scope("Downsample"):
            return downsample_2d(y, k = resample_kernel)

    # x is img_feats: the evolving image features grid, starting from the input resolution, and iteratively
    # processed and downsampled until making final binary prediction about the image source (real / fake).
    x = None

    # Main layers
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope("%dx%d" % (2**res, 2**res)):
            # Optional skip/resnet connections (see StyleGAN)
            if architecture == "skip" or res == resolution_log2:
                x = fromrgb(x, images_in, res)
            # Process features through the discriminator block
            x = block(x, res)
            # Downsampling for skip connection
            if architecture == "skip":
                images_in = downsample(images_in)

    # Final layers
    with tf.variable_scope("4x4"):
        if architecture == "skip":
            x = fromrgb(x, images_in, 2)
        # Minibatch standard deviation layer (see StyleGAN)
        if mbstd_group_size > 1:
            with tf.variable_scope("MinibatchStddev"):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope("Conv"):
            x = apply_bias_act(conv2d_layer(x, fmaps = nf(1), kernel = 3), act = act)
        # Turn final 4x4 image grid to a vector
        with tf.variable_scope("Dense0"):
            x = apply_bias_act(dense_layer(x, fmaps = nf(0)), act = act) # [batch_size, nf(0)]


    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope("Output"):
        x = apply_bias_act(dense_layer(x, fmaps = max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis = 1, keepdims = True) # [batch_size, 1]

    # Output
    scores_out = tf.identity(x, name = "scores_out")
    return scores_out # [batch_size, 1]


#----------------------------------------------------------------------------

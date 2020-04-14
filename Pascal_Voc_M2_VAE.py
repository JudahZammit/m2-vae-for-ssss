#!/usr/bin/env python
# coding: utf-8

# In[1]:


# M2_VAE for semantic segmentation
# A semi-supervised model for semantic segmentation on pascal voc


# In[2]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import keras
from keras.models import Model,load_model
from keras import layers
from keras.layers import (Input,Activation,Concatenate,Add,Dropout,BatchNormalization,Conv2D,DepthwiseConv2D
                        ,ZeroPadding2D,AveragePooling2D,Lambda,Conv2DTranspose, MaxPooling2D, concatenate
                        ,Dropout,UpSampling2D,Flatten)
from keras.engine import Layer,InputSpec
from keras.engine.topology import get_source_inputs
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

import pandas as pd

import math
import random
from random import randint

from sklearn.utils import class_weight

import albumentations as A

from PIL import Image

import cv2
import os
import glob

import tensorflow as tf
import tensorflow_probability as tfp

from keras import backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[3]:


#Shape of image: i.e. the image will be RESIZE X RESIZE
RESIZE = 128

#Number of possible classes for each pixel
CLASSES = 21
BATCH_SIZE = 16

# Used in the gumbel softmax sampling trick
TEMPERATURE = .1


# In[4]:


# Much of this code comes from 
# https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py
# however it has been heavily modified


# Functions and layers that are used by the deeplab networks
class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        #self.data_format = K.normalize_data_format(data_format)
        self.data_format = None
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] *                 input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] *                 input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False)(x)

    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate)
                      )(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate)
                      )(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None)(x)
        x = Activation(relu6)(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate))(x)

    x = Activation(relu6)(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None)(x)

    if skip_connection:
        return Add()([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


# In[5]:


def M2():

    input_shape = (RESIZE,RESIZE,3)
    alpha=1.
    img_input = Input(shape=input_shape)
    y_full = Input(shape=(RESIZE,RESIZE,CLASSES))
    y_input= Lambda(lambda x:  tf.split(x,num_or_size_splits=2,axis=0))(y_full)[0] 
    
    # A network that takes as input the given image and outputs the 
    # parameters to multinomial distributions for each pixel
    # i.e. a mask.
    # This network is refered to as q(y|x) in the paper
    OS = 8
    first_block_filters = _make_divisible(32 * alpha, 8)
    y = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False)(img_input)
    y = Activation(relu6)(y)

    y = _inverted_res_block(y, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    y = _inverted_res_block(y, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    y = _inverted_res_block(y, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    y = _inverted_res_block(y, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    y = _inverted_res_block(y, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    y = _inverted_res_block(y, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    y = _inverted_res_block(y, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    y = _inverted_res_block(y, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    y = _inverted_res_block(y, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    y = _inverted_res_block(y, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    y = _inverted_res_block(y, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    y = _inverted_res_block(y, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    y = _inverted_res_block(y, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    y = _inverted_res_block(y, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    y = _inverted_res_block(y, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    y = _inverted_res_block(y, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    y = _inverted_res_block(y, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(y)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False)(y)
    b0 = Activation('relu')(b0)

    # there are only 2 branches in mobilenetV2. not sure why

    y = Concatenate()([b4, b0])

    y = Conv2D(256, (1, 1), padding='same',
               use_bias=False)(y)
    y = Activation('relu')(y)
    y = Dropout(0 )(y)
    
    y = Conv2D(CLASSES, (1, 1), padding='same')(y)
    y = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(y)
    
    # This is the predicted mask for each image
    y_output = Activation('softmax',name = 'q_y')(y)
    
    # Splits out the masks that have a true mask(y_sup) and those that don't(y_un)
    y_sup,y_un =Lambda(lambda x:  tf.split(x,num_or_size_splits=2,axis=0))(y_output) 

    # A function that generates samples from a set of mulitnomial distributions 
    # in a way that the gradient can propagate through.
    def gumbel_softmax(args):
        ind_multinomial = args
        gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(TEMPERATURE, probs=ind_multinomial)
        return gumbel_dist.sample()
    
    # Samples from the "distribution" of the masks for the images without labels
    y_un_sample = Lambda(gumbel_softmax)(y_un)
    
    # Replaces the predicted masks for the images with labels with the true masks
    # this may seem wierd but it is what is mathematiclly correct
    y_t_un = Concatenate(axis=0)([y_input,y_un_sample])

    # END q(y|x)
    
    # A network that takes as input the half true half predicted masks as
    # well as the images as input and outputs the parameters to
    # a set of multivariate gaussian distributions
    # This network is refered to as q(z|y,x) in the paper
    ys = Concatenate(axis=-1)([y_t_un,img_input])

    OS = 8
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False)(ys)

    x = Activation(relu6)(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    b0 = Activation('relu')(b0)

    # there are only 2 branches in mobilenetV2. not sure why

    x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False)(x)
    x = Activation('relu')(x)
    x = Dropout(0)(x)

    # A log(sigma) for each latent variable
    # Note that the choices of CLASSES as the width of this output
    # is somewhat arbitrary
    # Note that log(sigma) instead of sigma or sigma^2 is chosen as the output for numericle stability
    z_log_var = Conv2D(CLASSES, (1, 1), padding='same')(x)
    z_log_var = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(z_log_var)
    
    # A mean for each latent variable
    z_mean = Conv2D(CLASSES, (1, 1), padding='same')(x)
    z_mean = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(z_mean)

    
    # A function for sampling from the above gaussian distrubution
    def gaussian_sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        z_sample = z_mean + K.exp(0.5 * z_log_var) * epsilon
        return z_sample
    
    # Samples form the predicted gaussian distribution
    z_sample = Lambda(gaussian_sampling,name = 'q_z')([z_mean, z_log_var])
    
    # END q(z|x,y)
    
    # A network that takes as input the above z sample and the
    # half true half predicted y and outputs the parameters to
    # a bernoulli distribution for each pixel and channel.
    # This could be interpruted as an image.
    # This is refered to as p(x|y,z) in the paper.
    x = Concatenate()([z_sample,y_t_un])
 
    input_shape = (RESIZE,RESIZE,CLASSES + CLASSES)

    OS = 8
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False)(x)
    x = Activation(relu6)(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    b0 = Activation('relu')(b0)

    # there are only 2 branches in mobilenetV2. not sure why

    x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False)(x)
    x = Activation('relu')(x)
    x = Dropout(0)(x)

    # DeepLab v.3+ decoder

    x = Conv2D(3, (1, 1), padding='same')(x)
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)

    x = Activation('sigmoid',name = 'p_x')(x)

    # END p(x|y,z)
    
    
    # A function that calcuates the intersection over union couf.
    def iou_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_input * y_sup), axis=[1,2,3])
        union = K.sum(y_input,[1,2,3])+K.sum(y_sup,[1,2,3])-intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou
    
    # Calculates the log liklihood of a point x under a gaussian distribution parameterized by mu and log_var
    def gaussian_ll(args):
        x , mu, log_var = args
        x = Flatten()(x)
        mu = Flatten()(mu)
        log_var = Flatten()(log_var)
        
        c = -.5 * math.log(2*math.pi)
        density = c - log_var/2 - ((x - mu)/(2*K.exp(log_var) + 1e-8))*(x - mu)

        return K.sum(density,axis = -1)
    
    # Calculates the log liklihood of a point x under a unit gaussian distribution
    def unit_gaussian_ll(args):
        x = args
        x = Flatten()(x)
        
        c = -.5 * math.log(2*math.pi)
        density = c - x**2/2

        return K.sum(density,axis = -1)

    
    # Calculates the log liklihood that the sampled 'z' is under the unit gaussian distributions 
    def log_pz(y_true,y_pred):
        loss = unit_gaussian_ll(z_sample)
        return loss
        
    
    # Calculates the log liklihood that the sampled 'z' is under the gaussian distributions predicted by
    # q(z|y,x)
    def log_qz(y_true,y_pred):
        loss = gaussian_ll([z_sample,z_mean,z_log_var])
        return loss
    
    # Calculates the log liklihood that the all possible 'y' is under y's true distribution WHICH
    # IS ASSUMED TO BE BERNOULLI WITH CONSTANT PROBABILITY 1/CLASSES FOR ALL Y
    def log_py(y_true,y_pred):
        y = Flatten()(y_t_un)
        ones = K.ones_like(y)/CLASSES
        loss = -K.binary_crossentropy(ones,y)
        loss = K.sum(loss,axis=1)
        return loss
    
    # Calculates the log liklihood that the true images is predicted by
    # p(x|y,z). Image is expected to be binarized.
    def log_px(y_true,y_pred):
        #Effectivly calculates
        #if(img_input == 1)
        #  loss = log(x)
        #else if(img_input == 0)
        #  loss = log(1 - x)
        loss = -K.binary_crossentropy(img_input,x)
        loss = K.sum(loss,axis = 1)
        return loss
    
    
    # Calculates the log liklihood that the sampled y is under the predicted y's distribution WHICH
    # IS ASSUMED TO BE BERNOULLI
    def log_qy(y_true,y_pred):
        zero = K.zeros(shape = ((BATCH_SIZE//2)))
        un = -K.binary_crossentropy(Flatten()(y_un),Flatten()(y_un_sample))
        un = K.sum(un,axis = 1)
        loss = K.concatenate([zero,un])
        return loss
       
    
    # Calculates a supervised loss for the y predictions for the images that we have labels for
    def y_class(y_true,y_pred):
        zero = K.zeros(shape = (BATCH_SIZE//2))
        sup_loss = K.binary_crossentropy(Flatten()(y_input),Flatten()(y_sup))
        sup_loss = K.sum(sup_loss,axis = 1)
        loss = K.concatenate([sup_loss,zero])
        return loss
    
    # Calculates the negative lower bounds(i.e. the minamization target) for q(y|x),q(z|x,y) and p(x|z,y)
    def y_loss(y_true,y_pred):
        return K.mean(1*log_qy(y_true,y_pred) + -1*log_py(y_true,y_pred) + 1000*y_class(y_true,y_pred))
    
    def z_loss(y_true,y_pred):
        return K.mean(log_qz(y_true,y_pred) + -1*log_pz(y_true,y_pred))
    
    def x_loss(y_true,y_pred):
        return K.mean(-log_px(y_true,y_pred))
    
    
    loss = {'p_x':x_loss,'q_z':z_loss,'q_y':y_loss}
    metrics = {'q_y':iou_coef}
    model = Model([img_input,y_full], [x,y_output,z_sample], name = 'VAE')
    model.compile(loss = loss,metrics = metrics,optimizer = keras.optimizers.Adam(lr=.1,clipnorm = 1.,clipvalue = 0.5))
    
    return model


# In[6]:


model = M2()
model.summary()


# In[7]:


#Defines the augmentitations
def get_training_augmentation():
    train_transform = [
        A.Flip(p=0.5),
        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),     
        A.Resize(height = RESIZE, width = RESIZE, interpolation=1, always_apply=True, p=1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        A.Resize(height = RESIZE, width = RESIZE, interpolation=1, always_apply=True, p=1)
    ]
    return A.Compose(test_transform)


# In[8]:


# Generates the data used for training
class TrainGenerator(keras.utils.Sequence):
    
    # Loads in unlabeled images(file paths) and repeats the labeled images until they're
    # are more labeled ones then unlabeled ones
    def __init__(self,  n_classes=21, batch_size=32, resize_shape=(RESIZE,RESIZE)):
        
        self.image_path_list = os.listdir('./VOCdevkit/VOC2012/train_frames/')
        self.unsupervised_path_list = os.listdir('./VOCdevkit/VOC2012/JPEGImages/')
        
        random.shuffle(self.image_path_list)
        random.shuffle(self.unsupervised_path_list)
          
        lis = os.listdir('./VOCdevkit/VOC2012/train_frames/')
        while len(self.image_path_list) <= len(self.unsupervised_path_list):
            random.shuffle(lis)
            self.image_path_list.extend(lis)
        
        self.n_classes = n_classes
        self.batch_size = int(batch_size/2)
        self.resize_shape = resize_shape
        if self.resize_shape:
            self.X = np.zeros((self.batch_size*2, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros((self.batch_size*2, resize_shape[1],resize_shape[0],n_classes), dtype='float32')
        else:
            raise Exception('No image dimensions specified!')
    
    # Number of epochs is number of unlabeled images divided by the batch size
    def __len__(self):
        return  len(self.unsupervised_path_list) // self.batch_size 
        
    # Fetches batch treating image as a matrix of the parameters 
    # to independent bernoulli distributed random variables, which are
    # then sampled from to create a dynamic discretization of the data.
    # Also dummy encodes the mask.
    def __getitem__(self, i):
        n = 0
        
        for x in self.image_path_list[i*self.batch_size:(i+1)*self.batch_size]:
            
            image = np.array(Image.open('./VOCdevkit/VOC2012/train_frames/' + x))
            label = np.array(Image.open('./VOCdevkit/VOC2012/train_masks/' + x.replace('.jpg','.png')))

            sample = get_training_augmentation()(image=image, mask=label)
            image, label = sample['image']/255,sample['mask']
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)
            categorical_label = keras.utils.to_categorical(label)

            self.X[n] = image
            #cat_label -> image
            self.Y[n] = categorical_label[:,:,0:21]
            n = n + 1
            
        for x in self.unsupervised_path_list[i*self.batch_size:(i+1)*self.batch_size]:
    
            image = np.array(Image.open('./VOCdevkit/VOC2012/JPEGImages/' + x))

            sample = get_training_augmentation()(image=image)
            image= sample['image']/255
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)

            self.X[n] = image
            n = n + 1

        return [self.X, self.Y] , [self.Y,self.Y,self.Y]
        
    def on_epoch_end(self):
        random.shuffle(self.unsupervised_path_list)
        self.image_path_list = os.listdir('./VOCdevkit/VOC2012/train_frames/')
        lis = os.listdir('./VOCdevkit/VOC2012/train_frames/')
        while len(self.image_path_list) <= len(self.unsupervised_path_list):
            random.shuffle(lis)
            self.image_path_list.extend(lis)  


# In[9]:


# Generates the data used for validation
class ValGenerator(keras.utils.Sequence):
    
    # Loads in the labeled images
    def __init__(self,  n_classes=21, batch_size=32, resize_shape=(RESIZE,RESIZE)):
            
        self.image_path_list = os.listdir('./VOCdevkit/VOC2012/val_frames/')
        random.shuffle(self.image_path_list)
        
        self.n_classes = n_classes
        self.batch_size = int(batch_size/2)
        self.resize_shape = resize_shape
        if self.resize_shape:
            self.X = np.zeros((self.batch_size*2, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros((self.batch_size*2, resize_shape[1],resize_shape[0],n_classes), dtype='float32')
        else:
            raise Exception('No image dimensions specified!')
        
    def __len__(self):
        return len(self.image_path_list) // self.batch_size
        
    def __getitem__(self, i):
        n = 0
        
        for x in self.image_path_list[i*self.batch_size:(i+1)*self.batch_size]:
            
            image = np.array(Image.open('./VOCdevkit/VOC2012/val_frames/' + x))
            label = np.array(Image.open('./VOCdevkit/VOC2012/val_masks/' + x.replace('.jpg','.png')))

            sample = get_validation_augmentation()(image=image, mask=label)
            image, label = sample['image']/255, sample['mask']
            rand = np.random.ranf(image.shape)
            image = np.greater(image,rand).astype(int)

            categorical_label = keras.utils.to_categorical(label)

            self.X[n] = image
            #cat_label -> image
            self.Y[n] = categorical_label[:,:,0:21]
            n = n + 1

        return [self.X, self.Y] , [self.Y,self.Y,self.Y]
        
    def on_epoch_end(self):
        random.shuffle(self.image_path_list)


# In[10]:


# Creates training and validition generators
train_gen = TrainGenerator(batch_size = BATCH_SIZE,n_classes = CLASSES)
val_gen = ValGenerator(batch_size = BATCH_SIZE,n_classes = CLASSES)


# In[11]:


# Saves the model best weights to a file 
checkpoint = ModelCheckpoint(
    'NEW_Pascal_Voc_M2_VAE.h5', 
    monitor='val_q_y_iou_coef', 
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False,
    mode='max',
    period = 1
)

# Reduces the learning rate when the model has stoped learning
reduce_lr = ReduceLROnPlateau(monitor='loss',patience = 3 ,factor = .5,verbose = 1)


# Trains the model for 10 epochs
history = model.fit_generator(
    generator = train_gen,
    validation_data=val_gen,
    callbacks=[checkpoint,reduce_lr],
    use_multiprocessing=False,
    workers=1,
    epochs=10 ,
    max_queue_size = 10
)


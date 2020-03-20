# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io
import pdb

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])#vggnet标准化

def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(data_path)#载入vgg.mat
    mean = data['normalization'][0][0][0]#获取标准化!!!
    mean_pixel = np.mean(mean, axis=(0, 1))#即[ 123.68 ,  116.779,  103.939]
    weights = data['layers'][0]#获取vgg训练好的用来提取图像特征的权重，即卷积核

    net = {}#字典
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        '''卷积层的操作'''
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]#对应layers每一层的权重值1
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width,i in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))# 即w   !!![height, width,i in_channels, out_channels]
            bias = bias.reshape(-1)#!!!b
            current = _conv_layer(current, kernels, bias)#进行卷积操作
        # relu层非线性的操作
        elif kind == 'relu':
            current = tf.nn.relu(current)
        # pooling层的操作
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current#键值对&conv1_1

    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')#补零进行卷积
    return tf.nn.bias_add(conv, bias)#加上偏置项


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')


def preprocess(image):#标准化
    return image - MEAN_PIXEL


def unprocess(image):
    return image + MEAN_PIXEL

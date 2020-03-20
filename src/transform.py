import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1

def net(image):
    #_conv_layer(net, num_filters, filter_size, strides, relu=True):
    conv1 = _conv_layer(image, 32, 9, 1)#卷积
    conv2 = _conv_layer(conv1, 64, 3, 2)#卷积
    conv3 = _conv_layer(conv2, 128, 3, 2)#卷积
    resid1 = _residual_block(conv3, 3)#残差网络
    resid2 = _residual_block(resid1, 3)#残差网络
    resid3 = _residual_block(resid2, 3)#残差网络
    resid4 = _residual_block(resid3, 3)#残差网络
    resid5 = _residual_block(resid4, 3)#残差网络
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)#反卷积
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)#反卷积
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)#卷积到num_filters=3
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2#映射值到0-255
    return preds
#卷积
def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)#初始化
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net
#反卷积
def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)#新的BHWC
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1,strides,strides,1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)

#残差
def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)

#标准化
def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keepdims=True)#均值方差
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

#权重初始化
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i for i in net.get_shape()]###i.value for i in net.get_shape()
    if not transpose:#卷积
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:#反卷积
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    weights_init = tf.Variable(tf.compat.v1.random.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init

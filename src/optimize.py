from __future__ import print_function
import functools
from src import vgg,transform
from src.utils import get_img
import pdb, time#!!!pdb debug
import tensorflow as tf, numpy as np, os
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):#!!!只进行两次前向？
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size#训练图像一次四张batch_size=4，训练集图片数量应为4的倍数
    if mod > 0:
        print("Train set has been trimmed slightly..")#训练集被轻微修剪
        content_targets = content_targets[:-mod] #去掉余数

    style_features = {}#!!!
    '''
    .shape=(HWC)
    style_shape=(1,图片垂直尺寸,图片水平尺寸,图片通道数)
    '''
    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    #BHWC
    print("所训练的style image属性(图片垂直尺寸/图片水平尺寸/图片通道数):"+style_target.shape)
    # precompute style features预计算的风格特征
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:#!!!cpu:0???
        '''
        tf.placeholder():
        dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
        shape：数据形状。默认是None，就是一维值，也可以是多维:(batch_size,图片垂直尺寸/图片水平尺寸/图片通道数)
        name：名称
        '''
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')#!!!session
        style_image_pre = vgg.preprocess(style_image)#1.将风格图标准化!!!图像-均值
        net = vgg.net(vgg_path, style_image_pre)#2.风格图标准化后并进入vgg
        style_pre = np.array([style_target])#风格图的矩阵形式
        '''
        取出vgg中过程的特征图，即不同阶段被卷积后的特征图
        '''
        for layer in STYLE_LAYERS:#取特定层
            features = net[layer].eval(feed_dict={style_image:style_pre})#喂style_pre给session即给style_image赋值为style_pre,拿到特征图
            features = np.reshape(features, (-1, features.shape[3]))#!!!
            gram = np.matmul(features.T, features) / features.size#计算gram值！！！！！！！！！！！！！
            style_features[layer] = gram
        '''
        取出vgg中过程的内容图，即relu4_2
        '''
    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")#一次四张内容图
        X_pre = vgg.preprocess(X_content)#标准化

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]#relu4_2   batch=4
        '''
        content_features:不经过生成网络的内容图
        preds_pre---net:经过生成网络的内容图，即中间图
        '''
        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)#归一化,float化，经过生产网络残差网络，也是batch=4
            preds_pre = vgg.preprocess(preds)#再经过vgg

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        '''
        Loss(Content)内容损失函数
        '''
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )
        '''
        Loss(Style)风格损失函数
        grams:经过生成网络的内容图，即中间图进行Gram
         style_gram:上面算过的不经过生成网络的特征图gram
        '''
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size#Loss(Style)风格损失函数

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
    #去噪loss值
        '''
        总的loss
        '''
        loss = content_loss + style_loss + tv_loss

        # overall loss
        '''
        梯度下降
        '''
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):#
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:#即一轮迭代
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):#每batch_size个即4个一组
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch
                }

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                '''
                判断到了设置的print_iterations轮数
                和判断是做完最后一轮迭代
                进行过程打印：should_print = is_print_iter or is_last
                '''
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                #打印
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)#保存迭代打印内容
                    yield(_preds, losses, iterations, epoch)#返回值

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

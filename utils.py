import tensorflow as tf
from scipy.misc import imresize
import numpy as np
from vgg19 import *
import scipy.stats as st

mean_RGB = np.array([123.68, 116.779, 103.939])


def resize_img(x, shape):
    x = np.copy(x).astype(np.uint8)
    y = imresize(x, shape, interp='bicubic')
    return y


def load_files(files, res, test_mode):
    if not test_mode:
        loaded = [scipy.misc.imread(filename, mode="RGB") for filename
                  in files]
    else:
        loaded = [scipy.misc.imread(filename, mode="RGB") for filename
                  in files]
    return loaded


def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def convlayer(input, output, ksize, stride, name, use_bn, activation="relu"):
    temp = tf.layers.conv2d(input, output, ksize, stride, padding="same", name=name,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    if use_bn:
        temp = tf.layers.batch_normalization(temp, name="BN_" + name)
    if activation:
        if activation == "relu":
            temp = tf.nn.relu(temp, name="relu_" + name)
        elif activation == "tanh":
            temp = tf.nn.tanh(temp, name="tanh_" + name)
        else:
            temp = activation(temp)
    return temp


def resblock(input, output, resblock_num, use_bn):
    rb_conv1 = convlayer(input, output, 3, 1, ("rb_%d_conv_1" % resblock_num), use_bn)
    rb_conv2 = convlayer(rb_conv1, output, 3, 1, ("rb_%d_conv_2" % resblock_num), use_bn)
    return rb_conv2 + input


def preprocess(img):
    return (img - mean_RGB) / 255


def postprocess(img):
    return np.round(np.clip(img*255 + mean_RGB, 0, 255)).astype(np.uint8)


def calc_pixel_loss(gt, generated):
    return tf.nn.l2_loss(gt - generated)


def get_content_loss(vgg_dir, gt, generated, content_layer):
    enhanced_vgg = net(vgg_dir, gt * 255)
    gt_vgg = net(vgg_dir, generated * 255)
    return tf.reduce_mean(tf.square(enhanced_vgg[content_layer] - gt_vgg[content_layer]))


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def gaussian_blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

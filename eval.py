import argparse
import os
from glob import glob
from scipy.misc import imread
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description="testing options")

parser.add_argument("phone_model", type=str, help="phone model to test")
parser.add_argument("--test_dir", type=str, default="D:/FYPdenoising/dslr/test")
parser.add_argument("--vgg_dir", type=str, help="directory for trained VGG 19 model",
                    default="vgg_pretrained/imagenet-vgg-verydeep-19.mat")
parser.add_argument("--content_layer", type=str, help="content layer to use in VGG 19 net", default="relu5_4")

config = parser.parse_args()

config.test_dir = os.path.join(config.test_dir, config.phone_model, "patches/*")

files = sorted(glob(config.test_dir))
num_samples = int(len(files) / 3)

gts = np.zeros((num_samples, 100, 100, 3), dtype=np.float32)
inputs = np.zeros((num_samples, 100, 100, 3), dtype=np.float32)
outputs = np.zeros((num_samples, 100, 100, 3), dtype=np.float32)
for i in range(num_samples):
    gts[i] = imread(files[i * 3], mode="RGB")
    inputs[i] = imread(files[i * 3 + 1], mode="RGB")
    outputs[i] = imread(files[i * 3 + 2], mode="RGB")

loss_output = tf.reduce_mean(tf.square(gaussian_blur(gts) - gaussian_blur(outputs)))
loss_input = tf.reduce_mean(tf.square(gaussian_blur(gts) - gaussian_blur(inputs)))
sess = tf.Session()
print("loss with output =", sess.run(loss_output))
print("loss with input =", sess.run(loss_input))

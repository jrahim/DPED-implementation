from utils import *
from vgg19 import *
import time
import os
import math
from matplotlib import pyplot as plt


class Model(object):
    def __init__(self, session, config, dataloader):
        self.sess = session
        self.config = config
        self.data_loader = dataloader
        self.noisy_train = dataloader.phone_data
        self.gt_train = dataloader.dslr_data

        self.generator_in = tf.placeholder(shape=[None, self.config.res, self.config.res, 3], dtype=tf.float32,
                                           name="generator_input")
        self.gt_in = tf.placeholder(shape=[None, self.config.res, self.config.res, 3], dtype=tf.float32,
                                    name="gt_input")
        self.enhanced_in = tf.placeholder(shape=[None, self.config.res, self.config.res, 3], dtype=tf.float32,
                                          name="enhanced_input")
        self.generator_in_test = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name="test_input")

        self.generated = self.generator(self.generator_in)
        self.discriminator_gt = self.discriminator(self.gt_in)
        self.discriminator_enhanced = self.discriminator(self.generated)

        self.generator_test = self.generator(self.generator_in_test)

        print("setting up loss functions")
        self.d_loss = -tf.reduce_mean(tf.log(self.discriminator_gt) + tf.log(1. - self.discriminator_enhanced))

        self.color_loss = 255 * tf.reduce_mean(tf.square(gaussian_blur(self.gt_in) - gaussian_blur(self.generated)))
        self.g_loss = self.config.w_adversarial_loss * -tf.reduce_mean(
            tf.log(
                self.discriminator_enhanced)) + self.config.w_pixel_loss * self.color_loss + self.config.w_content_loss * get_content_loss(
            self.config.vgg_dir, self.gt_in, self.generated,
            self.config.content_layer) + self.config.w_tv_loss * tf.reduce_mean(
            tf.image.total_variation(self.generated))

        t_vars = tf.trainable_variables()
        discriminator_vars = [var for var in t_vars if 'discriminator' in var.name]
        generator_vars = [var for var in t_vars if 'generator' in var.name]

        self.discriminator_solver = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.d_loss,
                                                                                               var_list=discriminator_vars)
        self.generator_solver = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.g_loss,
                                                                                           var_list=generator_vars)

        tf.global_variables_initializer().run(session=self.sess)
        self.saver = tf.train.Saver(tf.trainable_variables())

    def generator(self, feature_in):
        print("Setting up the generator network")
        use_bn = self.config.use_bn
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            conv1 = convlayer(feature_in, 64, 9, 1, "conv_1", use_bn)
            rb1 = resblock(conv1, 64, 1, use_bn)
            rb2 = resblock(rb1, 64, 2, use_bn)
            rb3 = resblock(rb2, 64, 3, use_bn)
            rb4 = resblock(rb3, 64, 4, use_bn)
            conv2 = convlayer(rb4, 64, 3, 1, "conv_2", use_bn)
            conv3 = convlayer(conv2, 64, 3, 1, "conv_3", use_bn)
            conv4 = convlayer(conv3, 64, 3, 1, "conv_4", use_bn)
            conv5 = convlayer(conv4, 3, 3, 1, "conv_5", False, activation=None)
            return conv5

    def discriminator(self, feature_in):
        print("setting up the discriminator network")
        use_bn = self.config.use_bn
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            in_gs = tf.image.rgb_to_grayscale(feature_in)
            conv1 = convlayer(in_gs, 48, 11, 5, "conv_1", use_bn, activation=None)
            conv1 = lrelu(conv1)
            conv2 = convlayer(conv1, 128, 5, 2, "conv_2", True, activation=None)
            conv2 = lrelu(conv2)
            conv3 = convlayer(conv2, 192, 3, 1, "conv_3", True, activation=None)
            conv3 = lrelu(conv3)
            conv4 = convlayer(conv3, 192, 3, 1, "conv_4", True, activation=None)
            conv4 = lrelu(conv4)
            conv5 = convlayer(conv4, 128, 3, 2, "conv_5", True, activation=None)
            conv5 = lrelu(conv5)
            flat = tf.contrib.layers.flatten(conv4)
            fc1 = tf.layers.dense(flat, units=1024, activation=None)
            fc1 = lrelu(fc1)
            logits = tf.layers.dense(fc1, units=1, activation=None)
            prob = tf.nn.sigmoid(logits)
            return prob

    def train(self, load=False):

        if load:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" Overall training starts from beginning")

        start = time.time()
        plt_epoch = []
        plt_loss = []
        for epoch in range(0, self.config.num_epochs):
            if epoch == 1:
                print("1 epoch completed")
            noisy_batch, gt_batch = self.data_loader.get_batch()
            _, enhanced_batch = self.sess.run([self.generator_solver, self.generated],
                                              feed_dict={self.generator_in: noisy_batch, self.gt_in: gt_batch})
            _ = self.sess.run(self.discriminator_solver,
                              feed_dict={self.generator_in: noisy_batch, self.gt_in: gt_batch})

            isnan = False
            if epoch % 200 == 0:
                g_loss = self.sess.run(self.g_loss,
                                       feed_dict={self.generator_in: noisy_batch, self.gt_in: gt_batch})
                print("Iteration %d, runtime: %.3f s, generator loss: %.6f" % (
                    epoch, time.time() - start, g_loss))
                if math.isnan(g_loss):
                    print("nan loss encountered, finishing training on epoch: ", epoch)
                    isnan = True
                else:
                    self.save()

            if isnan:
                break

            if epoch % 1000 == 0:
                plt_epoch.append(epoch)
                plt_loss.append(self.sess.run(self.g_loss,
                                              feed_dict={self.generator_in: noisy_batch, self.gt_in: gt_batch}))
                plt.clf()
                plt.plot(plt_epoch, plt_loss, color="blue")
                plt.title("Generator Loss vs. Epoch")
                plt.xlabel("Epoch")
                plt.ylabel("Generator Loss")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.checkpoint_dir, "graph.png"))
                self.save(epochnum=epoch)

    def test(self):
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return [], []
        enhanced_batch = []
        run_img = [[]]
        for i in range(len(self.noisy_train)):
            self.noisy_train[i] = preprocess(self.noisy_train[i])
            run_img[0] = self.noisy_train[i]
            enhanced_batch.extend(self.sess.run(self.generator_test, feed_dict={self.generator_in_test: run_img}))
        return self.noisy_train, enhanced_batch, self.gt_train

    def save(self, epochnum=None):
        checkpoint_dir = os.path.join(self.config.checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not epochnum:
            self.saver.save(self.sess, os.path.join(checkpoint_dir, self.config.phone_model), write_meta_graph=False)
        else:
            self.saver.save(self.sess, os.path.join(checkpoint_dir, self.config.phone_model + str(epochnum)),
                            write_meta_graph=False)

    def load(self):
        checkpoint_dir = os.path.join(self.config.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("Loading checkpoints from ", checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if not self.config.epoch_to_load:
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, self.config.phone_model))
            else:
                self.saver.restore(self.sess, os.path.join(checkpoint_dir,
                                                           self.config.phone_model + str(self.config.epoch_to_load)))
            return True
        else:
            return False

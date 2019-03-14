# coding=utf-8
import tensorflow as tf
from GAN.ArtGAN import ArtGAN
from common.common import getImg, torch_decay, getFiles, saveImg, imgPool, encode, resizeTo
import numpy as np
import random
import threading, os, time, cv2
from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('input',
                           "D://100p04000000081fwD834.jpg",
                           'The directory that input pictures are saved')
tf.app.flags.DEFINE_string('output', "out/11.jpg", 'The directory that validate pictures are saved')
tf.app.flags.DEFINE_string('checkpoint', "model/model_ps.ckpt",
                           'The directory that trained network will be saved')
tf.app.flags.DEFINE_string('Norm', 'INSTANCE', 'Choose to use BatchNorm or instanceNorm')
tf.app.flags.DEFINE_integer('img_size', 768, 'The size of input img')
tf.app.flags.DEFINE_integer('ngf', 32, 'The number of gen filters in generater layer')
tf.app.flags.DEFINE_integer('ndf', 64, 'The number of gen filters in discriminator layer')
tf.app.flags.DEFINE_integer('batch_size', 1, 'The batch size of testing')

FLAGS = tf.app.flags.FLAGS


def main():
    start = time.time()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        content_img = resizeTo(getImg(FLAGS.input), FLAGS.img_size)
        content = np.expand_dims(encode(content_img), 0)
        network = ArtGAN(FLAGS.batch_size, FLAGS.ngf, FLAGS.img_size, is_training=True, Norm=FLAGS.Norm)
        network.test(content.shape)
        sess.run(tf.global_variables_initializer())
        var_list = [var for var in tf.global_variables() if
                    'generator_net' in var.name or 'discriminator_net' in var.name]
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, FLAGS.checkpoint)
        print("restored all")
        s = time.time()
        out = sess.run(network.output, feed_dict={network.content: content})
        # result = np.clip(out[0], 0, 255).astype(np.uint8)
        print("Transform in {} s".format((time.time() - s)))
        saveImg(out[0], FLAGS.output)
        print("Finished all process in {} s".format(time.time() - start))


if __name__ == '__main__':
    main()

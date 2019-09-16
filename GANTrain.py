# coding=utf-8
import tensorflow as tf
from GAN.ArtGAN import ArtGAN
from common.common import getImg, torch_decay, getFiles, saveImg, imgRandomCrop, encode, linear_decay, resizeTo
from Preprocessing.image_preprocessing import Augmentor
import numpy as np
import random
import threading, os, time, cv2

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('img_content', "D:/train_large_places365standard",
                           'The directory that content pictures are saved')
tf.app.flags.DEFINE_string('img_style', "D:/pablo-picasso",
                           'The directory that style picture are saved')
tf.app.flags.DEFINE_string('checkpoint', "checkout/",
                           'The directory that trained network will be saved')
tf.app.flags.DEFINE_string('Norm', 'INSTANCE', 'Choose to use Batchnorm or instanceNorm')
tf.app.flags.DEFINE_float('learning_rate', 2e-4, 'The init learning rate')
tf.app.flags.DEFINE_float('decay', 1e-6, 'The init learning rate decay')
tf.app.flags.DEFINE_integer('start_step', 299999, 'The start step for linear decay')
tf.app.flags.DEFINE_integer('end_step', 300000, 'The end step for linear decay')
tf.app.flags.DEFINE_integer('max_to_keep', 10, 'The maximum ckpt num')
tf.app.flags.DEFINE_integer('summary_iter', 10, 'The steps per summary')
tf.app.flags.DEFINE_integer('save_iter', 200, 'The steps per save')
tf.app.flags.DEFINE_integer('batch_size', 1, 'The batch size of training')
tf.app.flags.DEFINE_integer('multi_threads', 5, 'The number of threads used')
tf.app.flags.DEFINE_float('discr', 1.0, 'The weight of D_loss')
tf.app.flags.DEFINE_float('img', 100.0, 'The weight of img_loss')
tf.app.flags.DEFINE_float('feature', 100.0, 'The weight of feature_loss')
tf.app.flags.DEFINE_integer('ngf', 32, 'The number of gen filters in generater layer')
tf.app.flags.DEFINE_integer('ndf', 64, 'The number of gen filters in discriminator layer')
tf.app.flags.DEFINE_float('win_rate', 0.8, 'The value used to choose to train dis or gen')
tf.app.flags.DEFINE_integer('img_size', 768, 'The size of input img')
tf.app.flags.DEFINE_integer('together_step', 0, 'The number of steps that d and g are trained together')

FLAGS = tf.app.flags.FLAGS

augmentor = Augmentor(crop_size=[FLAGS.img_size, FLAGS.img_size],
                      vertical_flip_prb=0.,
                      hsv_augm_prb=1.0,
                      hue_augm_shift=0.05,
                      saturation_augm_shift=0.05, saturation_augm_scale=0.05,
                      value_augm_shift=0.05, value_augm_scale=0.05, )

files_content = getFiles(FLAGS.img_content, 'content')
files_style = getFiles(FLAGS.img_style, 'style')

def generateBatch(files, batch_shape):
    batch = np.zeros(batch_shape, dtype=np.float32)
    while True:
        try:
            choosed = random.sample(files, batch_shape[0])
            for i, s in enumerate(choosed):
                batch[i] = augmentor(resizeTo(getImg(s), 800, 1800))
                batch[i] = encode(batch[i])
            yield batch
        except:
            continue


def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
        queue_content = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
        queue_style = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
        queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32, tf.float32],
                             shapes=[[FLAGS.img_size, FLAGS.img_size, 3], [FLAGS.img_size, FLAGS.img_size, 3]])
        enqueue_op = queue.enqueue_many([queue_content, queue_style])
        dequeue_op = queue.dequeue()
        content_batch_op, style_batch_op = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=100)

        with tf.device('/device:CPU:0'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.where(tf.greater_equal(global_step, FLAGS.start_step),
                                     linear_decay(FLAGS.learning_rate, global_step, FLAGS.start_step, FLAGS.end_step),
                                     FLAGS.learning_rate)
            # learning_rate = torch_decay(FLAGS.learning_rate, global_step1, FLAGS.decay)
            opt1 = tf.train.AdamOptimizer(learning_rate)
            opt2 = tf.train.AdamOptimizer(learning_rate)

        net = ArtGAN(FLAGS.batch_size, FLAGS.ngf, FLAGS.ndf, FLAGS.img_size, FLAGS.Norm)
        net.train(FLAGS.discr, FLAGS.img, FLAGS.feature)

        var_list_1 = [var for var in tf.trainable_variables() if
                      'Encoder_Model' in var.name or 'Decoder_Model' in var.name]
        var_list_2 = [var for var in tf.trainable_variables() if 'Disc_Model' in var.name]

        if "BATCH" in FLAGS.Norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_1 = opt1.minimize(net.T_Loss, global_step, var_list_1)
                train_op_2 = opt2.minimize(net.D_Loss, global_step, var_list_2)
                train_op = tf.group([train_op_1, train_op_2])
        else:
            train_op_1 = opt1.minimize(net.T_Loss, global_step, var_list_1)
            train_op_2 = opt2.minimize(net.D_Loss, global_step, var_list_2)
            train_op = tf.group([train_op_1, train_op_2])

        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

        with tf.device('/device:CPU:0'):
            with tf.name_scope('summary'):
                tf.summary.scalar('learning_rate', learning_rate)
                with tf.name_scope('gen_img'):
                    tf.summary.image('output', net.output)
                    tf.summary.image('content', net.content)
                    tf.summary.image('style', net.style)
                with tf.name_scope('loss'):
                    tf.summary.scalar('D_Loss', net.D_Loss)
                    tf.summary.scalar('G_Loss', net.G_Loss)
                    tf.summary.scalar('img_Loss', net.img_Loss)
                    tf.summary.scalar('feature_Loss', net.feature_Loss)
                    tf.summary.scalar('T_Loss', net.T_Loss)
                with tf.name_scope('acc'):
                    tf.summary.scalar('D_Acc', net.D_Acc)
                    tf.summary.scalar('G_Acc', net.G_Acc)
                summary_op = tf.summary.merge_all()

        coord = tf.train.Coordinator()

        def enqueue(sess):
            img_content = generateBatch(files_content, (FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
            img_style = generateBatch(files_style, (FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
            while not coord.should_stop():
                content_batch = next(img_content)
                style_batch = next(img_style)
                try:
                    sess.run(enqueue_op, feed_dict={queue_content: content_batch, queue_style: style_batch})
                except:
                    print("The img reading thread is end")

        log_path = os.path.join(FLAGS.checkpoint, 'log')
        summary_writer = tf.summary.FileWriter(log_path)
        sess.run(tf.global_variables_initializer())

        if os.path.exists(os.path.join(FLAGS.checkpoint, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))

        with tf.device('/device:CPU:0'):
            iteration = global_step.eval() + 1

        enqueue_thread = []
        for i in range(FLAGS.multi_threads):
            enqueue_thread.append(threading.Thread(target=enqueue, args=[sess]))
            enqueue_thread[i].isDaemon()
            enqueue_thread[i].start()

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        discr_success = FLAGS.win_rate

        while True:
            try:
                start = time.time()
                content_batch, style_batch = sess.run([content_batch_op, style_batch_op])
                if iteration <= FLAGS.together_step:
                    output = sess.run(
                        {'global_step': global_step, 'learning_rate': learning_rate, 'train_op': train_op,
                         'output': net.output, 'D_Acc': net.D_Acc, 'G_Acc': net.G_Acc, 'summary': summary_op,
                         'D_Loss': net.D_Loss, 'G_Loss': net.G_Loss, 'img_Loss': net.img_Loss,
                         'feature_Loss': net.feature_Loss, 'T_Loss': net.T_Loss},
                        feed_dict={net.content: content_batch, net.style: style_batch})
                    string = 'Train together'
                elif discr_success >= FLAGS.win_rate:
                    output = sess.run(
                        {'global_step': global_step, 'learning_rate': learning_rate, 'train_op': train_op_1,
                         'output': net.output, 'D_Acc': net.D_Acc, 'G_Acc': net.G_Acc, 'summary': summary_op,
                         'D_Loss': net.D_Loss, 'G_Loss': net.G_Loss, 'img_Loss': net.img_Loss,
                         'feature_Loss': net.feature_Loss, 'T_Loss': net.T_Loss},
                        feed_dict={net.content: content_batch, net.style: style_batch})
                    string = 'Train G'
                    discr_success = discr_success * (1. - 0.05) + 0.05 * (1 - output['G_Acc'])
                else:
                    output = sess.run(
                        {'global_step': global_step, 'learning_rate': learning_rate, 'train_op': train_op_2,
                         'output': net.output, 'D_Acc': net.D_Acc, 'G_Acc': net.G_Acc, 'summary': summary_op,
                         'D_Loss': net.D_Loss, 'G_Loss': net.G_Loss, 'img_Loss': net.img_Loss,
                         'feature_Loss': net.feature_Loss, 'T_Loss': net.T_Loss},
                        feed_dict={net.content: content_batch, net.style: style_batch})
                    string = 'Train D'
                    discr_success = discr_success * (1. - 0.05) + 0.05 * output['D_Acc']

            except Exception as e:
                coord.request_stop(e)
                print("Get error as {} , need reload".format(e))
                if os.path.exists(os.path.join(FLAGS.checkpoint, 'checkpoint')):
                    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))
                    print("Restoring checkpoint")
                    continue
                else:
                    print("No checkpoint")
                    break

            if iteration % FLAGS.summary_iter == 0:
                summary_writer.add_summary(output['summary'], output['global_step'])

            if iteration % FLAGS.save_iter == 0:
                save_path = saver.save(sess, os.path.join(FLAGS.checkpoint, 'model.ckpt'), output['global_step'])
                print("Model saved in file: %s" % save_path)

            print(
                "At Step {},with learning_rate is {:.7f}, get D_Loss {:.2f}, G_Loss {:.2f}, img_Loss {:.2f}, "
                "feature_Loss {:.2f}, T_Loss {:.2f}, D_Acc {:.2f}, G_Acc {:.2f}, success {:.3f}, cost {:.2f}s,"
                "{:s}".
                    format(
                    output['global_step'],
                    output['learning_rate'],
                    output['D_Loss'],
                    output['G_Loss'],
                    output['img_Loss'],
                    output['feature_Loss'],
                    output['T_Loss'],
                    output['D_Acc'],
                    output['G_Acc'],
                    discr_success,
                    time.time() - start,
                    string)
            )

            if (output['global_step'] >= FLAGS.end_step):
                break

            iteration += 1
        print('done')
        save_path = saver.save(sess, os.path.join(FLAGS.checkpoint, 'model.ckpt'), output['global_step'])
        print("Model saved in file: %s" % save_path)
        coord.request_stop()
        queue.close()
        coord.join(threads)
        print("All end")


if __name__ == '__main__':
    train()

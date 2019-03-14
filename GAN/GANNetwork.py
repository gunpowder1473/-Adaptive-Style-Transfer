import net.net as net
import tensorflow as tf


class Encoder:
    def __init__(self, name, ngf, Norm='NOT', is_training=True):
        self.ngf = ngf
        self.name = name
        self.reuse = False
        self.Norm = Norm
        self.is_training = is_training

    def __call__(self, image):
        with tf.variable_scope('generator_net' + self.name, reuse=self.reuse):
            # result = tf.pad(image, [[0, 0], [18, 18], [18, 18], [0, 0]], "REFLECT")
            # pre = net.convLayer(result, 3, 1, strides=1, Norm='NOT', training=self.is_training,
            #                     name='Edge_P', pad='VALID', relu=False)
            # result1 = net.convLayer(pre, 1, 3, strides=1, Norm=self.Norm, training=self.is_training,
            #                         name='Edge_1', pad='VALID', relu='RELU')
            # result2 = net.convLayer(pre, 1, 3, strides=1, Norm=self.Norm, training=self.is_training,
            #                         name='Edge_2', pad='VALID', relu='RELU')
            # result = net.convLayer(result1 + result2, 3, 3, strides=1, Norm=self.Norm, training=self.is_training,
            #                        name='Edge_3', pad='VALID', relu=False)
            # result = tf.sigmoid(result) * 2 - 1
            if 'BATCH' in self.Norm:
                result = net.batchNorm(image, self.is_training, 'first_Norm')
            elif 'INSTANCE' in self.Norm:
                result = net.instanceNorm(image, 'first_Norm', self.is_training, )
            else:
                result = image
            result = tf.pad(result, [[0, 0], [15, 15], [15, 15], [0, 0]], "REFLECT")
            result = net.convLayer(result, self.ngf, 3, strides=1, Norm=self.Norm, training=self.is_training,
                                   name='3x3_1', pad='VALID', relu='RELU')
            result = net.convLayer(result, self.ngf, 3, strides=2, Norm=self.Norm, training=self.is_training,
                                   name='3x3_2', pad='VALID', relu='RELU')
            result = net.convLayer(result, 2 * self.ngf, 3, strides=2, Norm=self.Norm, training=self.is_training,
                                   name='3x3_3', pad='VALID', relu='RELU')
            result = net.convLayer(result, 4 * self.ngf, 3, strides=2, Norm=self.Norm, training=self.is_training,
                                   name='3x3_4', pad='VALID', relu='RELU')
            result = net.convLayer(result, 8 * self.ngf, 3, strides=2, Norm=self.Norm, training=self.is_training,
                                   name='3x3_5', pad='VALID', relu='RELU')

        self.reuse = True
        return result


class Decoder:
    def __init__(self, name, ngf, Norm='NOT', is_training=True):
        self.ngf = ngf
        self.name = name
        self.reuse = False
        self.Norm = Norm
        self.is_training = is_training

    def __call__(self, image):
        with tf.variable_scope('generator_net' + self.name, reuse=self.reuse):
            num = image.get_shape().as_list()[-1]
            result = image
            for i in range(9):
                result = net.residualBlock(result, num, 3, 'res' + '{:d}'.format(i), relu='RELU', pad='REFLECT',
                                           training=self.is_training, Norm=self.Norm)
            result = net.resizeConv2D(result, 8 * self.ngf, 3, name='r3x3_1', strides=2, pad='SAME', relu='RELU',
                                      training=self.is_training, Norm=self.Norm)
            result = net.resizeConv2D(result, 4 * self.ngf, 3, name='r3x3_2', strides=2, pad='SAME', relu='RELU',
                                      training=self.is_training, Norm=self.Norm)
            result = net.resizeConv2D(result, 2 * self.ngf, 3, name='r3x3_3', strides=2, pad='SAME', relu='RELU',
                                      training=self.is_training, Norm=self.Norm)
            result = net.resizeConv2D(result, self.ngf, 3, name='r3x3_4', strides=2, pad='SAME', relu='RELU',
                                      training=self.is_training, Norm=self.Norm)
            result = tf.nn.sigmoid(net.convLayer(result, 3, 7, name='7x7_1', strides=1, pad='REFLECT', relu=False,
                                                 training=self.is_training, Norm='NOT')) * 2. - 1.
            # result = tf.nn.tanh(net.convLayer(result, 3, 7, name='7x7_1', strides=1, pad='REFLECT', relu=False,
            #                                   training=self.is_training, Norm='NOT'))

        self.reuse = True
        return result


class Discriminator:
    def __init__(self, name, ndf, Norm=False, is_training=True):
        self.reuse = False
        self.Norm = Norm
        self.ndf = ndf
        self.is_training = is_training
        self.name = name

    def __call__(self, image):
        with tf.variable_scope('discriminator_net' + self.name, reuse=self.reuse):
            result = net.convLayer(image, self.ndf * 2, 5, strides=2, name='5x5_1', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result_1 = net.convLayer(result, 1, 5, strides=1, name='r1', relu=False, Norm='NOT',
                                     training=self.is_training, pad='SAME')
            result = net.convLayer(result, self.ndf * 2, 5, strides=2, name='5x5_2', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result_2 = net.convLayer(result, 1, 10, strides=1, name='r2', relu=False, Norm='NOT',
                                     training=self.is_training, pad='SAME')
            result = net.convLayer(result, self.ndf * 4, 5, strides=2, name='5x5_3', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result = net.convLayer(result, self.ndf * 8, 5, strides=2, name='5x5_4', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result_4 = net.convLayer(result, 1, 10, strides=1, name='r4', relu=False, Norm='NOT',
                                     training=self.is_training, pad='SAME')
            result = net.convLayer(result, self.ndf * 8, 5, strides=2, name='5x5_5', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result = net.convLayer(result, self.ndf * 16, 5, strides=2, name='5x5_6', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result_6 = net.convLayer(result, 1, 6, strides=1, name='r6', relu=False, Norm='NOT',
                                     training=self.is_training, pad='SAME')
            result = net.convLayer(result, self.ndf * 16, 5, strides=2, name='5x5_7', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result_7 = net.convLayer(result, 1, 3, strides=1, name='r7', relu=False, Norm='NOT',
                                     training=self.is_training, pad='SAME')

        self.reuse = True
        return result_1, result_2, result_4, result_6, result_7


class Edge:
    def __init__(self, name, Norm='INSTANCE', is_training=True):
        self.reuse = False
        self.Norm = Norm
        self.is_training = is_training
        self.name = name

    def __call__(self, image):
        with tf.variable_scope('edge_net' + self.name, reuse=self.reuse):
            pre = net.convLayer(image, 3, 1, strides=1, Norm='NOT', training=self.is_training,
                                name='Edge_P', relu=False)
            result1 = net.convLayer(pre, 1, 3, strides=1, Norm=self.Norm, training=self.is_training,
                                    name='Edge_1', pad='REFLECT', relu='RELU')
            result2 = net.convLayer(pre, 1, 3, strides=1, Norm=self.Norm, training=self.is_training,
                                    name='Edge_2', pad='REFLECT', relu='RELU')
            result = net.convLayer(result1 + result2, 3, 3, strides=1, Norm='NOT', training=self.is_training,
                                   relu=False, name='Edge_3')
            result = tf.tanh(result)
        self.reuse = True
        return result

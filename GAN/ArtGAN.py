from GAN import GANNetwork
import tensorflow as tf
import common.common as cm
from tensorflow.contrib import slim


class ArtGAN:
    def __init__(self, batch_size, ngf=32, ndf=64, img_size=128, Norm='INSTANCE', is_training=True):
        self.img_size = img_size
        self.batch_size = batch_size
        self.win_rate = win_rate
        self.Encoder = GANNetwork.Encoder('Encoder_Model', ngf, Norm=Norm, is_training=is_training)
        self.Decoder = GANNetwork.Decoder('Decoder_Model', ngf, Norm=Norm, is_training=is_training)
        self.Disc = GANNetwork.Discriminator('Disc_Model', ndf, Norm=Norm, is_training=is_training)
        self.Edge = GANNetwork.Edge('Edge_Model', Norm=Norm, is_training=is_training)

    def transferBlock(self, input, kernel_size=10):
        return slim.avg_pool2d(inputs=input, kernel_size=kernel_size, stride=1, padding='SAME')

    def dgLoss(self, logits, labels):
        # loss = tf.squared_difference(logits, labels)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(loss)

    def imageLoss(self, input, target):
        return tf.reduce_mean((input - target) ** 2)

    def featureLoss(self, input, target):
        return tf.reduce_mean(tf.abs(input - target))

    def train(self, discr, img, feature):
        self.content = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size, self.img_size, 3),
                                      name='img_content')
        self.style = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size, self.img_size, 3),
                                    name='img_style')

        self.content_feature = self.Encoder(self.content)
        self.output = self.Decoder(self.content_feature)
        self.output_feature = self.Encoder(self.output)

        self.content_pre = self.Disc(self.content)
        self.style_pre = self.Disc(self.style)
        self.output_pre = self.Disc(self.output)

        transfered_output = self.transferBlock(self.output)
        transfered_input = self.transferBlock(self.content)

        self.content_Dloss = []
        self.style_Dloss = []
        self.output_Dloss = []
        self.content_Dacc = []
        self.style_Dacc = []
        self.output_Dacc = []

        self.G_Loss = []
        self.G_Acc = []

        for i in self.content_pre:
            self.content_Dloss.append(self.dgLoss(i, tf.zeros_like(i)))
            self.content_Dacc.append(tf.reduce_mean(tf.cast(x=(i < tf.zeros_like(i)), dtype=tf.float32)))
        for i in self.style_pre:
            self.style_Dloss.append(self.dgLoss(i, tf.ones_like(i)))
            self.style_Dacc.append(tf.reduce_mean(tf.cast(x=(i > tf.zeros_like(i)), dtype=tf.float32)))
        for i in self.output_pre:
            self.output_Dloss.append(self.dgLoss(i, tf.zeros_like(i)))
            self.output_Dacc.append(tf.reduce_mean(tf.cast(x=(i < tf.zeros_like(i)), dtype=tf.float32)))
            self.G_Loss.append(self.dgLoss(i, tf.ones_like(i)))
            self.G_Acc.append(tf.reduce_mean(tf.cast(x=(i > tf.zeros_like(i)), dtype=tf.float32)))

        with tf.name_scope('losses'):
            self.D_Loss = discr * (
                    tf.add_n(self.content_Dloss) + tf.add_n(self.style_Dloss) + tf.add_n(self.output_Dloss))
            self.D_Acc = (tf.add_n(self.content_Dacc) + tf.add_n(self.style_Dacc) + tf.add_n(
                self.output_Dacc)) / 3. / float(len(self.content_pre))
            self.G_Loss = discr * tf.add_n(self.G_Loss)
            self.G_Acc = tf.add_n(self.G_Acc) / float(len(self.output_pre))
            self.img_Loss = img * self.imageLoss(transfered_output, transfered_input)
            self.feature_Loss = feature * self.featureLoss(self.output_feature, self.content_feature)
            self.T_Loss = self.G_Loss + self.img_Loss + self.feature_Loss


    def test(self, batch_shape):
        self.content = tf.placeholder(dtype=tf.float32, shape=batch_shape,
                                      name='img_content')
        self.output = cm.valid(self.Decoder(self.Encoder(self.content)))

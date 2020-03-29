from utils import *
import tensorflow as tf
import numpy as np
from loadData import DataSet
import random


class GANNet(object):

    def __init__(self, category='Mnist', vis_step=10, Train_Epochs=200, batch_size=128, z_size=100, lr=0.001,
                 history_dir='./', checkpoint_dir='./', logs_dir='./', gen_dir='./'):
        self.test = False

        self.category = category
        self.epoch = Train_Epochs
        self.img_size = 28 if (category == 'Fashion-Mnist' or category == 'Mnist') else 64
        self.batch_size = batch_size
        self.z_size = z_size
        self.vis_step = vis_step

        self.lr = lr
        self.channel = 1 if (category == 'Fashion-Mnist' or category == 'Mnist') else 3

        self.history_dir = history_dir
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.gen_dir = gen_dir

        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_size], name='latent')

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, self.channel],
                                name='image')


    def build_Model(self):
        self.gen = self.Generator(self.z, reuse=False)
        self.D_x = self.Discriminator(self.x, reuse=False)
        self.D_gen = self.Discriminator(self.gen, reuse=True)

        """
        Loss and Optimizer
        """
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_x, labels=tf.ones_like(self.D_x)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_gen, labels=tf.zeros_like(self.D_gen)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_gen, labels=tf.ones_like(self.D_gen)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_var = [var for var in tf.trainable_variables() if var.name.startswith('Gen')]
        self.d_var = [var for var in tf.trainable_variables() if var.name.startswith('Dis')]

        self.d_optim = tf.train.AdamOptimizer(self.lr) \
            .minimize(self.d_loss, var_list=self.d_var)
        self.g_optim = tf.train.AdamOptimizer(self.lr) \
            .minimize(self.g_loss, var_list=self.g_var)
        """
        Logs
        """
        tf.summary.scalar('g_loss', tf.reduce_mean(self.g_loss))
        tf.summary.scalar('d_loss', tf.reduce_mean(self.d_loss))
        # TODO showing specifically
        # tf.summary.histogram('hyper params', self.hyper_var)
        self.summary_op = tf.summary.merge_all()


    def Generator(self, z, reuse=False):
        with tf.variable_scope('Gen', reuse=reuse):
            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':
                z = tf.reshape(z, [-1, self.z_size])

                fc1 = tf.layers.dense(inputs=z, units=1024, name='fc1')

                fc1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(fc1, is_training=True))

                fc2 = tf.layers.dense(inputs=fc1, units=6272, name='fc2')

                fc2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(fc2, is_training=True))

                fc2 = tf.reshape(fc2, [self.batch_size, 7, 7, 128])

                dc1 = deconv2d(fc2, (self.batch_size, self.img_size // 2, self.img_size // 2, 128), kernal=(5, 5),
                               name='dc1')
                dc1 = tf.contrib.layers.batch_norm(dc1, is_training=True)
                dc1 = tf.nn.leaky_relu(dc1)

                dc2 = deconv2d(dc1, (self.batch_size, self.img_size // 1, self.img_size // 1, 1), kernal=(5, 5),
                               name='dc2')

                output = tf.nn.tanh(dc2)

            return output

    def Discriminator(self, x, reuse=False):
        with tf.variable_scope('Dis', reuse=reuse):
            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':

                c1 = tf.nn.leaky_relu(conv2d(x, 1, name='c1'))

                c2 = tf.contrib.layers.batch_norm(conv2d(c1, 64, name='c2'), is_training=True)
                c2 = tf.nn.leaky_relu(c2)
                c2 = tf.reshape(c2, [self.batch_size, -1])

                fc1 = tf.layers.dense(inputs=c2, units=1024, name='fc1')
                fc1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(fc1, is_training=True))

                output = tf.layers.dense(inputs=fc1, units=1, name='fc2')

            return output

    def train(self, sess):
        self.build_Model()

        data = DataSet(img_size=self.img_size, batch_size=self.batch_size, category=self.category)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        writer = tf.summary.FileWriter(self.logs_dir, sess.graph)

        start = 0
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint:
            latest_checkpoint.split('-')
            start = int(latest_checkpoint.split('-')[-1])
            saver.restore(sess, latest_checkpoint)
            print('Loading checkpoint {}.'.format(latest_checkpoint))

        tf.get_default_graph().finalize()

        # latent_gen = np.random.normal(size=(len(data), self.z_size))

        for epoch in range(start + 1, self.epoch):
            num_batch = int(len(data) / self.batch_size)
            d_losses = []
            g_losses = []
            for step in range(num_batch):
                obs = data.NextBatch(step)
                # z = latent_gen[step * self.batch_size: (step + 1) * self.batch_size].copy()
                z = np.random.normal(size=(self.batch_size, self.z_size))

                d_loss, _ = sess.run([self.d_loss, self.d_optim], feed_dict={self.z: z, self.x: obs})
                d_losses.append(d_loss)
                # writer.add_summary(summary, global_step=epoch)

                g_loss, _ = sess.run([self.g_loss, self.g_optim], feed_dict={self.z: z})
                g_losses.append(g_loss)
                # writer.add_summary(summary, global_step=epoch)

                # _ = sess.run(self.g_optim, feed_dict={self.z: z})


            print(epoch, " dis Loss: ", np.mean(d_losses), " gen loss: ", np.mean(g_losses))
            if epoch % self.vis_step == 0:
                self.visualize(saver, sess, len(data), epoch, np.random.normal(size=(len(data), self.z_size)), data)

    def visualize(self, saver, sess, num_data, epoch, latent_gen, data):
        saver.save(sess, "%s/%s" % (self.checkpoint_dir, 'model.ckpt'), global_step=epoch)
        idx = random.randint(0, int(num_data / self.batch_size) - 1)
        z = latent_gen[idx * self.batch_size: (idx + 1) * self.batch_size]

        """
        Generation
        """
        # obs = data.NextBatch(idx, test=True)
        z = np.random.normal(size=(self.batch_size, self.z_size))
        sys = sess.run(self.gen, feed_dict={self.z: z})
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.gen_dir + 'epoch' + str(epoch) + 'gens.jpg'
        show_in_one(path, sys, column=16, row=8)

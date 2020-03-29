# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:
from GAN_net import GANNet
import tensorflow as tf
import os
import shutil
from save import saveModule

FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_integer('vis_step', 200, 'Epoch when visualize')
tf.flags.DEFINE_integer('Train_Epochs', 2000, 'Number of Epochs to train')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size of training images')
tf.flags.DEFINE_integer('z_size', 200, 'dimensions of latent factors')

tf.flags.DEFINE_float('lr', 0.000, 'learning rate')

tf.flags.DEFINE_string('category', 'Mnist', 'DataSet category')

tf.flags.DEFINE_string('history_dir', './output/history/', 'history')
tf.flags.DEFINE_string('checkpoint_dir', './output/checkpoint/', 'checkpoint')
tf.flags.DEFINE_string('logs_dir', './output/logs/', 'logs')
tf.flags.DEFINE_string('gen_dir', './output/gens/', 'gen')


def main(_):
    saveFlag = True
    saveFlag = False
    if saveFlag:
        save = saveModule(category=FLAGS.category,
                          z_size=FLAGS.z_size,
                          lr=FLAGS.lr)
        save.process()

    else:
        model = GANNet(
            category=FLAGS.category,
            vis_step=FLAGS.vis_step,
            Train_Epochs=FLAGS.Train_Epochs,
            batch_size=FLAGS.batch_size,
            z_size=FLAGS.z_size,
            lr=FLAGS.lr,
            history_dir=FLAGS.history_dir,
            checkpoint_dir=FLAGS.checkpoint_dir,
            logs_dir=FLAGS.logs_dir,
            gen_dir=FLAGS.gen_dir,
        )

        continueTrain = False
        # continueTrain = True
        with tf.Session() as sess:
            if not continueTrain:
                if os.path.exists(FLAGS.checkpoint_dir):
                    shutil.rmtree(FLAGS.checkpoint_dir[:-1])
                os.makedirs(FLAGS.checkpoint_dir)

            if os.path.exists(FLAGS.logs_dir):
                shutil.rmtree(FLAGS.logs_dir[:-1])
            os.makedirs(FLAGS.logs_dir)

            if not os.path.exists(FLAGS.history_dir):
                os.makedirs(FLAGS.history_dir)


            if os.path.exists(FLAGS.gen_dir):
                shutil.rmtree(FLAGS.gen_dir[:-1])
            os.makedirs(FLAGS.gen_dir)

            model.train(sess)


if __name__ == '__main__':
    tf.app.run()

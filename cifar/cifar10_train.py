import argparse
import datetime
import sys
import time

import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning

import cifar10_pruning as cifar10

FLAGS = None


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        images, labels = cifar10.distorted_inputs()

        logits = cifar10.inference(images)
        loss = cifar10.loss(logits, labels)

        train_op = cifar10.train(loss, global_step)
        pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)
        pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)
        mask_update_op = pruning_obj.conditional_mask_update_op()
        pruning_obj.add_pruning_summaries()

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = 128
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print(
                        format_str % (datetime.datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                                                      tf.train.NanTensorHook(loss), _LoggerHook()],
                                               config=tf.ConfigProto(
                                                   log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                # Update the masks
                mon_sess.run(mask_update_op)


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/cifar10_train',
        help='Directory where to write event logs and checkpoint.')
    parser.add_argument(
        '--pruning_hparams',
        type=str,
        default='',
        help="""Comma separated list of pruning-related hyperparameters""")
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000000,
        help='Number of batches to run.')
    parser.add_argument(
        '--log_device_placement',
        type=bool,
        default=False,
        help='Whether to log device placement.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

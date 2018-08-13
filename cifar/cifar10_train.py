import argparse
import datetime
import sys
import time

import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning

import model as cifar10

FLAGS=None

def train():
    with tf.Graph().as_default():
        global_step=tf.contrib.framework.get_or_create_global_step()
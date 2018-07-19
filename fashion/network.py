import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from data import load_data

BATCH_SIZE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1


def model(x_train, y_train):
    train_data_node = tf.placeholder(tf.float16, shape=(BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE))
    train_labels_node = tf.placeholder(tf.float16, shape=(BATCH_SIZE,10))
    inputs = np.arange(28 * 28, dtype=np.float)
    label = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    inputs = tf.reshape(inputs, [1, 28, 28, 1])
    print(train_data_node.shape)
    train_data_node=tf.layers.flatten(train_data_node)
    net = fully_connected(inputs=train_data_node, num_outputs=128)
    print(net.shape)
    net = fully_connected(inputs=net, num_outputs=10)
    print(net.shape)
    prediction = tf.nn.softmax(net)
    print(prediction.shape)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(train_labels_node * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(300):
            batch_data = x_train[i:i+10, ...]
            batch_labels = y_train[i:i+10]
            batch_labels=tf.one_hot(batch_labels,10)
            print(sess.run(train_step, feed_dict={train_data_node: batch_data, train_labels_node: batch_labels}))


y_train, x_train, y_test, x_test = load_data()
print(x_train.dtype)
model(x_train, y_train)


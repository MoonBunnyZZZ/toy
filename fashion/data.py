import gzip

import numpy
import numpy as np
import tensorflow as tf

files = [
    'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
]


def load_data():
    with gzip.open('./datasets/' + files[0], 'rb') as lbpath:
        print(type(lbpath))
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open('./datasets/' + files[1], 'rb') as lbpath:
        x_train = np.frombuffer(lbpath.read(), np.uint8, offset=16).reshape(len(y_train), 28 * 28)
        x_train = x_train / 255
    with gzip.open('./datasets/' + files[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open('./datasets/' + files[3], 'rb') as lbpath:
        x_test = np.frombuffer(lbpath.read(), np.uint8, offset=16).reshape(len(y_test), 28 * 28)
        x_test = x_test / 255
    return y_train, x_train, y_test, x_test


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
      f: A file object that can be passed into a gzip reader.

    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


# with gfile.Open('/home/moon/project/frame_advanced/tf/fashion/datasets/t10k-labels-idx1-ubyte.gz', 'rb') as f:
#     train_images = extract_images(f)
def test_graph():
    g2 = tf.Graph()

    # 在计算图g1中定义张量和操作
    with tf.Graph().as_default() as g1:
        a = tf.constant([1.0, 1.0])
        b = tf.constant([1.0, 1.0])
        result1 = a + b

    with g2.as_default():
        a = tf.constant([2.0, 2.0])
        b = tf.constant([2.0, 2.0])
        result2 = a + b

    # 在g1计算图上创建会话
    with tf.Session(graph=g1) as sess:
        out = sess.run(result1)
        print('with graph g1, result: {0}'.format(out))

    with tf.Session(graph=g2) as sess:
        out = sess.run(result2)
        print('with graph g2, result: {0}'.format(out))

    print(g1.version)


def load_mnist(path, kind='train'):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


images, labels = load_mnist('./datasets')

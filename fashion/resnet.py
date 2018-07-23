import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


def batch_norm(inputs, training, data_format):
    return tf.layers.batch_normalization(inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
                                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                         scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channel_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                            kernel_initializer=tf.variance_scaling_initializer(), data_format=data_format)


def _building_block_v1(inputs,filters,training,projection_shortcut,strides,data_format):
    shortcut=inputs
    if projection_shortcut is not None:
        short=projection_shortcut(inputs)

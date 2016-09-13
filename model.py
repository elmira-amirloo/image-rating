import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, input_placeholder):
        self.input_placeholder = input_placeholder

    @staticmethod
    def load_trained_model(data_path, session, skip_layer=None):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            if key not in skip_layer:
                with tf.variable_scope(key, reuse=True):
                    for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                        session.run(tf.get_variable(subkey).assign(data))

    @classmethod
    def _conv_layer(cls, input_placeholder, kernel, stride, padding):
        return tf.nn.conv2d(input_placeholder, kernel, [1, stride, stride, 1], padding=padding)


    @classmethod
    def _conv_relu(cls, input_placeholder, kernel_size, stride, num_out ,name, padding='SAME', group=1):
        num_in = input_placeholder.get_shape()[-1]
        if num_in%group!=0 or num_out%group!=0:
            raise ValueError('the dimension of output/input conv layer is not compatible with number of groups.')

        with tf.variable_scope(name) as scope:
            kernel = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_in/group, num_out])
            biases = tf.get_variable('biases', [num_out])
            if group==1:
                conv = cls._conv_layer(input_placeholder, kernel, stride, padding)
            else:
                input_groups = tf.split(3, group, input_placeholder)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [cls._conv_layer(i, k, stride, padding) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            return tf.nn.relu(bias, name=scope.name)

    @classmethod
    def _relu(cls, input_placeholder, name):
        return tf.nn.relu(input_placeholder, name=name)

    @classmethod
    def _max_pool(cls, input_placeholder, kernel_size, stride, name):
        return tf.nn.max_pool(input_placeholder,
                              ksize=[1, kernel_size, kernel_size, 1],
                              strides=[1, stride, stride, 1],
                              padding='VALID',
                              name=name)

    @classmethod
    def _lrn(cls, input_placeholder, name):
        return tf.nn.local_response_normalization(input_placeholder, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name=name)


    @classmethod
    def _fc(cls, input_placeholder, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out])
            biases = tf.get_variable('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(input_placeholder, weights, biases, name=scope.name)
            return fc

    @classmethod
    def _dropout(cls, input_placeholder, keep_prob):
        return tf.nn.dropout(input_placeholder, keep_prob)


    def get_network(self, _var):
        network = self._conv_relu(self.input_placeholder, 11, 4, 96, padding='VALID', name='conv1')
        network = self._max_pool(network, 3, 2, name='pool1')
        network = self._lrn(network, name='norm1')

        network = self._conv_relu(network, 5, 1, 256, group=2, name='conv2')
        network = self._max_pool(network, 3, 2, name='pool2')
        network = self._lrn(network, name='norm2')

        network = self._conv_relu(network, 3, 1, 384, name='conv3')

        network = self._conv_relu(network, 3, 1, 384, group=2, name='conv4')

        network = self._conv_relu(network, 3, 1, 256, group=2, name='conv5')
        network = self._max_pool(network, 3, 2, name='pool5')

        network = tf.reshape(network, [-1, 6*6*256])
        network = self._fc(network, 6*6*256, 4096, name='fc6')
        network = self._dropout(network, _var)

        network = self._fc(network, 4096, 4096, name='fc7')
        network = self._dropout(network, _var)

        network = self._fc(network, 4096, 5, relu=False, name='fc8')
        return network

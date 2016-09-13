import sys
import json
import os

import cv2
import tensorflow as tf
import numpy as np

from model import Model

import logging
logger = logging.getLogger('training')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class Classifier(object):

    def __init__(self, classifier_directory, train=True):
        """Initializing classifier with model definition and parameters"""
        params_path = os.path.join(classifier_directory, 'params.json')
        if not os.path.exists(params_path):
            raise EnvironmentError('The model parameters are not provided in the specified location: %s', params_path)
        with open (params_path) as input_file:
            self.params = json.load (input_file)

        self.classifier_directory = classifier_directory

        #training parameters
        self.training_set = []
        self.test_set = []
        self.train_batch_start = 0
        self.test_batch_start = 0

        #detection parameters
        self.model_weights = None


    def _fit_img_to_model(self, img_path, scale_size):
        """ Read the model and adjust it accordingly to the model"""
        if not os.path.exists(img_path):
            raise IOError('img does not exist')
        img = cv2.imread(img_path)
        if img is not None:
            h, w, c = np.shape(img)
            resize_to = (scale_size, scale_size)
            img = cv2.resize(img, resize_to)
            img = img.astype(np.float32)
            img = img[None, ...]
            return img
        else:
            raise IOError('Failed to open img.')


    def _get_next_img_path(self, train=True):
        if train:
            img_path = self.train_set[self.train_batch_start]['path']
            img_clss = self.train_set[self.train_batch_start]['class']
            if self.train_batch_start < len(self.train_set):
                self.train_batch_start += 1
            else:
                self.train_batch_start = 0

        else:
            img_path = self.test_set[self.test_batch_start]['path']
            img_clss = self.test_set[self.test_batch_start]['class']
            if self.test_batch_start < len(self.test_set):
                self.test_batch_start += 1
            else:
                self.test_batch_start = 0
        return img_path, clss

    def _get_data_batch(self, batch_size, train=True):

        images = np.ndarray([batch_size, self.params["crop_size"], self.params["crop_size"], 3])
        one_hot_label = np.zeros((batch_size, self.params["n_classes"]))

        imgs_list = []
        if train:
            if self.train_batch_start + batch_size < len(self.training_set):
                imgs_list = self.training_set[self.train_batch_start: self.train_batch_start + batch_size]
                self.train_batch_start += batch_size
            else:
                next_starting_point = (self.train_batch_start + batch_size)%self.len(self.training_set)
                imgs_list = self.training_set[self.train_batch_start:] + self.training_set[:next_starting_point]
                self.train_batch_start = next_starting_point
        else:
            if self.test_batch_start + batch_size < len(self.test_set):
                imgs_list = self.test_set[self.test_batch_start: self.test_batch_start + batch_size]
                self.test_batch_start += batch_size
            else:
                next_starting_point = (self.test_batch_start + batch_size)%self.len(self.test_set)
                imgs_list = self.test_set[self.test_batch_start:] + self.test_set[:next_starting_point]
                self.test_batch_start = next_starting_point

        for indx, img in enumerate(imgs_list):
            if indx == batch_size:
                break
            is_img_readable = False
            counter = 0
            img_path = img['path']
            img_class = img['class']
            while not is_img_readable:
                try:
                    img = self._read_image(img_path)
                    is_img_readable = True
                except IOError:
                    logging.warning('img : {} is not added to training_set'.format(img['path']))
                    img_path, img_class = self._get_next_img_path(train)
                    counter += 1
                    if counter > len(imgs_list):
                        logger.fatal('provided training images are not readable')
                        break
            images[indx] = img
            one_hot_label[indx][img_class] = 1
        return images, one_hot_label


    def _read_image(self, img_path, mean_training=np.array([0., 0., 0.])):
        """reading the img"""
        #cv2 usually doesn't throw an error properly while reading img. this
        #function will handle it.
        if not os.path.exists(img_path):
            raise IOError('img does not exist')
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (self.params["scale_size"], self.params["scale_size"]))
            crop_size = self.params["crop_size"]
            img = img.astype(np.float32)
            img -= mean_training
            h, w, c = img.shape
            ho, wo = ((h-crop_size)/2, (w-crop_size)/2)
            img = img[ho:ho+crop_size, wo:wo+crop_size, :]
            return img
        else:
            raise IOError('Failed to open img.')


    @classmethod
    def _read_data_list(cls, data_path):
        data_list = []
        with open(data_path) as data_file:
            lines = data_file.readlines()
            for l in lines:
                items = l.split()
                data_list.append({ "path": items[0],
                                   "class": int(items[1])
                                })
        return data_list


    def train(self, train_list_path, test_list_path, learning_rate=0.001,
              training_iters=1470, batch_size =50, test_step=70, save_step=10, keep_rate=0.5
            ):

        def get_test_accuracy(sess, accuracy, number_of_test_batches):
            accuracy_test_set = 0.
            test_num = 0
            while test_num < number_of_test_batches:
                batch_tx, batch_ty = self._get_data_batch(batch_size, train=False)
                accuracy_each_test = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, _var: 1.})
                accuracy_test_set += accuracy_each_test
                test_num += 1
            accuracy_test_set /= test_num
            return accuracy_test_set


        """ Main function for training """
        self.training_set = self._read_data_list(train_list_path)
        self.test_set = self._read_data_list(test_list_path)
        n_classes = self.params['n_classes']
        number_of_test_batches = int(len(self.test_set)/batch_size)
        x = tf.placeholder(tf.float32, [batch_size, self.params['crop_size'], self.params['crop_size'], 3])
        y = tf.placeholder(tf.float32, [None, self.params['n_classes']])
        _var = tf.placeholder(tf.float32)

        model = Model(x)
        nn = model.get_network(_var)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(nn, y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        correct_pred = tf.equal(tf.argmax(nn, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        accuracy_training = []
        loss_training = []
        accuracy_testing = []

        with tf.Session() as sess:
            logger.info('Initializing variables...')
            sess.run(init)
            Model.load_trained_model(os.path.join(self.classifier_directory, 'pretrained.npy'), sess, ['fc8'])

            logger.info('Start training...')
            iter_num = 0
            while iter_num < training_iters:
                batch_xs, batch_ys = self._get_data_batch(batch_size)
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, _var: keep_rate})

                if iter_num%test_step == 0:
                    test_accuracy = get_test_accuracy(sess, accuracy, number_of_test_batches)
                    logger.info("iteration: {0}, test accuracy is {1}".format(iter_num, test_accuracy))
                    accuracy_testing.append({'step':iter_num, 'acc': test_accuracy})

                if iter_num%save_step == 0:
                    current_batch_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, _var: 1.})
                    current_batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, _var: 1.})
                    logger.info('Iteration: {0}, training accuracy: {1}, loss:{2}'.format(iter_num, current_batch_accuracy, current_batch_loss))
                    saver.save(sess, "{0}/saved_model_{1}.ckpt".format(self.classifier_directory, str(iter_num)))
                    accuracy_training.append({'step':iter_num, 'acc': current_batch_accuracy})

                iter_num += 1
        return accuracy_training, accuracy_testing


    def rate_image(self, img_path):
        """ The main function for img rating """
        if self.model_weights is None:
            self.model_weights = os.path.join(self.classifier_directory, 'model_weights.ckpt')
            if not os.path.exists(params_path):
                raise EnvironmentError('The model parameters are not provided in the specified location: %s', params_path)

        tf.reset_default_graph() #making sure the memory is released
        img_input = self._fit_img_to_model(img_path, self.params['crop_size'])

        with tf.Graph().as_default() as g:
            with tf.Session() as sess:
                x = tf.placeholder(tf.float32, [1, self.params['crop_size'], self.params['crop_size'], 3])
                y = tf.placeholder(tf.float32, [None, self.params['n_classes']])
                _var = tf.placeholder(tf.float32)
                # Model
                model_arch = Model.alexnet(x, _var)
                saver = tf.train.Saver()
                saver.restore(sess, self.model_weights)
                prob = tf.nn.softmax(model_arch)
                output = sess.run(prob, feed_dict={x:img_input, _var:1.})
        return output

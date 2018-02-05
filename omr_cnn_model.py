# _*_ utf-8 _*_

import os
import numpy as np
import random
import tensorflow as tf


def get_dataset():
    # get omr data
    data = DataSet()
    data.dataset_files = ['./tf_card1_1216.tfrecords',
                          './tf_card2_1216.tfrecords',
                          './tf_card3_1216.tfrecords']
    data.read_data()
    return data


def make_model(data):
    # training model
    model = Model()
    model.set_model_path_name('./', 'm2018_1')
    model.train_model(data)
    return model


def model_test(model_path_name, dataset):
    # use model
    model_app = CnnApp()
    model_app.load_model(model_path_name)
    model_app.load_model()
    model_app.test(dataset)
    model_app.predict(dataset[0][0:10])
    return model_app


def model_predict(model_path_name, imageset):
    # use model
    model_app = CnnApp()
    model_app.load_model(model_path_name)
    model_app.load_model()
    model_app.predict(imageset[0:10])
    return model_app


class DataSet:
    def __init__(self):
        self.name = 'dataset_omr123_testset'
        self.office_dataset_files = ['f:/studies/juyunxia/tfdata/tf_card1_1216.tfrecords',
                                     'f:/studies/juyunxia/tfdata/tf_card2_1216.tfrecords',
                                     'f:/studies/juyunxia/tfdata/tf_card3_1216.tfrecords']
        self.dataset_files = self.office_dataset_files

        # dataset para
        self.example_len = 1000
        self.train_rate = 0.85
        self.train_len = int(self.example_len * self.train_rate)
        self.test_len = self.example_len - self.train_len
        self.image_shape = (12, 16)

        # dataset with images, labels
        # [image_array, label_array]
        # iamge data normlized by 1/255, label one-hot
        self.data_set = [[], []]

    def read_data(self):
        # read data from tfrecords
        self.data_set = [[], []]
        for ds in self.dataset_files:
            print('read %s' % ds)
            rd = self.fun_read_tfrecord_tolist(ds)
            if len(rd) > 0:
                self.data_set[0] = self.data_set[0] + rd[0]
                self.data_set[1] = self.data_set[1] + rd[1]
            else:
                print('no data read in %s' % ds)
        if len(self.data_set) > 0:
            self.example_len = len(self.data_set[0])
            self.train_len = int(self.example_len * self.train_rate)
            self.test_len = self.example_len - self.train_len
            self.image_shape = self.data_set[0][0].shape
            self.blend_data()

    def blend_data(self):
        do = [x for x in range(len(self.data_set[0]))]
        random.shuffle(do)
        res_data = [[self.data_set[0][x] for x in do],
                    [self.data_set[1][x] for x in do]
                   ]

    @staticmethod
    def fun_read_tfrecord_tolist(tf_data_file):
        # get image, label data from tfrecord file
        # labellist = [lableitems:int, ... ]
        # imagedict = [label:imagematix, ...]
        if not os.path.isfile(tf_data_file):
            print(f'file error: not found file: \"{tf_data_file}\"!')
            return [[], []]
        count = 0
        image_list = []
        label_list = []
        example = tf.train.Example()
        for serialized_example in tf.python_io.tf_record_iterator(tf_data_file):
            example.ParseFromString(serialized_example)
            image = example.features.feature['image'].bytes_list.value
            label = example.features.feature['label'].bytes_list.value
            # precessing
            # img = np.zeros([image_shape[0] * image_shape[1]])
            # for i in range(len(img)):
            #    img[i] = image[0][i]
            img = np.array([image[0][x] for x in range(len(image[0]))])
            image_list.append(img / 255)
            labelvalue = int(chr(label[0][0]))
            label_list.append((1, 0) if labelvalue == 0 else (0, 1))
            count += 1
            if count % 3000 == 0:
                print('read record:%5d' % count)
        print(f'total images= {count}')
        return [image_list, label_list]

    def get_train_data(self, batchnum, starting_location):
        # read batchnum data in [0, train_len]
        if batchnum > self.train_len:
            batchnum = self.train_len
        if (starting_location + batchnum) > self.train_len:
            starting_location = 0
        # print(f'train examples {callnum}')
        res_data = [self.data_set[0][starting_location: starting_location + batchnum],
                    self.data_set[1][starting_location: starting_location + batchnum]]
        return res_data

    def get_test_data(self):
        res_data = [self.data_set[0][self.train_len: self.train_len+self.test_len],
                    self.data_set[1][self.train_len: self.train_len + self.test_len]]
        # res_data = [[self.data_set[0][i]
        #             for i in range(self.train_len, self.train_len+self.test_len)],
        #            self.data_set[1][self.train_len:self.train_len + self.test_len]]
        return res_data


class Model:

    def __init__(self):
        self.model_path = './'
        self.model_name = 'omr_model'
        self.save_model_path_name = './m18test'

    def set_model_path_name(self, path, name):
        self.model_path = path
        self.model_name = name
        self.save_model_path_name = self.model_path + self.model_name

    @staticmethod
    def weight_var(shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial_value=init)

    @staticmethod
    def bias_var(shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_value=init)

    @staticmethod
    def conv2d(x, w):
        res = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        return res

    @staticmethod
    def max_pool_2x2(x):
        res = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
        return res

    def train_model(self, data: DataSet, batch_num=40, train_num=1000):
        """ use dataset to train model"""

        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        # laye--input
        x = tf.placeholder(tf.float32, [None, 192], name='input_omr_images')
        y_ = tf.placeholder(tf.float32, [None, 2], name='input_omr_labels')
        x_image = tf.reshape(x, [-1, 12, 16, 1])

        # layer--conn-1
        w_conv1 = self.weight_var([4, 6, 1, 32])
        b_conv1 = self.bias_var([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, w_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # layer--conn-2 (hidden)
        w_conv2 = self.weight_var([4, 4, 32, 128])
        b_conv2 = self.bias_var([128])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # layer--fullconnect-1
        w_fc1 = self.weight_var([3*4*128, 256])
        b_fc1 = self.bias_var([256])
        h_pool1_flat = tf.reshape(h_pool2, [-1, 3*4*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # layer--fullconnect-2, output
        w_fc2 = self.weight_var([256, 2])
        b_fc2 = self.bias_var([2])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

        # model setting
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                                      reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # model running
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        tf.add_to_collection('predict_label', y_conv)
        tf.add_to_collection('accuracy', accuracy)
        # print(f'trainning data={data.name}')
        for i in range(train_num):
            batch = data.get_train_data(batch_num, i * 20)
            if i % 50 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0
                })
                print("step=%4d,\t accuracy= %2.8f" % (i, train_accuracy), '\t',
                      'cross_entropy=%1.10f' % cross_entropy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                      )
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

        omr_test_data = data.get_test_data()
        print('test accuracy= %1.6f' % accuracy.eval(
            feed_dict={
                x: omr_test_data[0], y_: omr_test_data[1], keep_prob: 1.0
            }
        ))
        saver.save(sess, self.save_model_path_name+'.ckpt')

    # omr_dataset: [(block_image, block_label), ...,]
    def use_model(self, omr_dataset):
        modelmeta = self.save_model_path_name + '.ckpt.meta'
        modelckpt = self.save_model_path_name + '.ckpt'
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(modelmeta)
        with tf.Session() as sess:
            saver.restore(sess, modelckpt)
            y = tf.get_collection('predict_label')[0]
            a = tf.get_collection('accuracy')[0]
            graph = tf.get_default_graph()
            # y 有placeholder "input_omr_images"，
            # sess.run(y)的时候还需要用实际待预测的样本
            # 以及相应的参数(keep_porb)来填充这些placeholder，
            # 这些需要通过graph的get_operation_by_name方法来获取。
            input_x = graph.get_operation_by_name('input_omr_images').outputs[0]
            input_y = graph.get_operation_by_name('input_omr_labels').outputs[0]
            keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
            # 使用 y 进行预测
            yp = sess.run(y, feed_dict={input_x: omr_dataset[0], keep_prob: 1.0})
            ac = sess.run(a, feed_dict={input_x: omr_dataset[0],
                                        input_y: omr_dataset[1],
                                        keep_prob: 1.0})
        return yp, ac


class CnnApp:

    def __init__(self):
        self.default_model_path_name = './omrmodel'
        self.model_path_name = ''
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.saver = None
        self.input_x = None
        self.input_y = None
        self.keep_prob = None
        self.y = None
        self.a = None

    def __del__(self):
        self.sess.close()

    def load_model(self, _model_path_name=''):
        if len(_model_path_name) == 0:
            self.model_path_name = self.default_model_path_name
        else:
            self.model_path_name = _model_path_name
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(self.model_path_name + '.ckpt.meta')
            self.saver.restore(self.sess, self.model_path_name+'.ckpt')
            self.y = tf.get_collection('predict_label')[0]
            self.a = tf.get_collection('accuracy')[0]
            self.input_x = self.graph.get_operation_by_name('input_omr_images').outputs[0]
            self.input_y = self.graph.get_operation_by_name('input_omr_labels').outputs[0]
            self.keep_prob = self.graph.get_operation_by_name('keep_prob').outputs[0]
            # yp = self.sess.run(self.y, feed_dict={self.input_x: omr_image_set, self.keep_prob: 1.0})
            # return yp

    def test(self, omr_data_set):
        with self.graph.as_default():
            # 测试, 计算识别结果及识别率
            yp = self.sess.run(self.y, feed_dict={self.input_x: omr_data_set[0], self.keep_prob: 1.0})
            ac = self.sess.run(self.a, feed_dict={self.input_x: omr_data_set[0],
                                                  self.input_y: omr_data_set[1],
                                                  self.keep_prob: 1.0})
            print(f'accuracy={ac}')
            yr = [(1 if v[0] < v[1] else 0, i) for i, v in enumerate(yp)]
            #     if [1 if v[0] < v[1] else 0][0] != omr_data_set[1][1]]
            err = [v1 for v1, v2 in zip(yr, omr_data_set[1]) if v1[0] != v2[1]]
        return err

    def predict(self, omr_image_set):
        with self.graph.as_default():
            # 使用 y 进行预测
            yp = self.sess.run(self.y, feed_dict={self.input_x: omr_image_set, self.keep_prob: 1.0})
        return yp

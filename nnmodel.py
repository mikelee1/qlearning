import tensorflow as tf
import pandas as pd
import numpy as np


class model:
    def __init__(self, actions, statenum):
        self.actions = actions
        self.statenum = statenum
        self.sess = tf.Session()
        self.x = tf.placeholder(shape=[None, 2], dtype=np.float32)
        self.y = tf.placeholder(shape=[None, 2], dtype=np.float32)

        self.w1 = tf.Variable(tf.ones(shape=[2, 2]), dtype=np.float32)
        self.logits = tf.nn.relu(tf.matmul(self.x, self.w1))
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.logits), reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(1).minimize(self.loss)
        self.initiation = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.timer = 0
        self.ckptdir = '/home/mike/Downloads/reinforelearn/qlearning_nn/ckpt/'
        if tf.train.get_checkpoint_state(self.ckptdir):
            self.saver.restore(self.sess, save_path=self.ckptdir)

    def train(self, trainlist):
        a = pd.DataFrame(trainlist)
        a['d'] = np.nan
        trainlist = a.values.tolist()
        self.timer += 1
        data = pd.DataFrame(trainlist, columns=['a', 'b', 'c', 'd'])
        data = data.drop_duplicates()
        data.loc[data.c == 'left', ['c', 'd']] = [0, 1]
        data.loc[data.c == 'right', ['c', 'd']] = [1, 0]

        x_data = np.array(data.iloc[:, :2], dtype=np.float32)
        y_data = np.array(data.iloc[:, [2, 3]], dtype=np.float32)
        self.trainlist = x_data
        self.y_data = y_data

        self.sess.run(self.initiation)
        self.sess.run(self.optimizer, feed_dict={self.x: x_data, self.y: y_data})
        if self.timer % 500 == 0:
            print('epoch: ', self.timer)
            self.saver.save(self.sess, save_path=self.ckptdir)
            print(self.sess.run(tf.argmax(self.sess.run(self.logits, feed_dict={self.x: [[2, 3]]}), 1)))
            print(self.sess.run(tf.argmax(self.sess.run(self.logits, feed_dict={self.x: [[4, 3]]}), 1)))

    def evaluate(self, state):
        a = self.sess.run(self.logits, feed_dict={self.x: state})
        #print(self.sess.run(self.w1))
        return self.sess.run(tf.argmax(a, 1))

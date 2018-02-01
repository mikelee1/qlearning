import tensorflow as tf
import pandas as pd
import numpy as np

def generate_data(x):
    x1 = [x,x+1,1,0]
    x2 = [x+1,x,0,1]
    return [x1,x2]



class model:
    def __init__(self,actions,statenum):
        self.dict = {'left':-1,'right':1}
        self.actions = actions
        self.statenum = statenum
        self.sess = tf.Session()
        self.x = tf.placeholder(shape=[None,2],dtype=np.float32)
        self.y = tf.placeholder(shape=[None,2],dtype=np.float32)

        self.bias1 = tf.Variable(np.random.random(),dtype=np.float32)
        self.layer1 = tf.Variable(tf.ones(shape=[2,2]),dtype=np.float32)
        self.logits = tf.nn.relu(tf.matmul(self.x, self.layer1))
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.logits),reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(1).minimize(self.loss)
        self.initiation = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.timer = 0


    def train(self,trainlist):
        a  = pd.DataFrame(trainlist)
        a['d']=np.nan
        trainlist = a.values.tolist()
        self.timer +=1
        data =pd.DataFrame(trainlist,columns=['a','b','c','d'])
        data = data.drop_duplicates()
        data.loc[data.c == 'left',['c','d']]=[0,1]
        data.loc[data.c == 'right',['c','d']] =[1, 0]

        x_data = np.array(data.iloc[:,:2],dtype=np.float32)
        y_data = np.array(data.iloc[:,[2,3]],dtype=np.float32)
        self.trainlist = x_data
        self.y_data = y_data

        self.sess.run(self.initiation)
        #print(self.sess.run(self.layer2))
        self.sess.run(self.optimizer,feed_dict={self.x:x_data,self.y:y_data})
        if self.timer %500 ==0:
            print('epoch: ',self.timer)
            print(self.sess.run(self.layer1))

            self.saver.save(self.sess, save_path='/home/mike/Downloads/reinforelearn/qlearning_nn/')
            print(self.sess.run(self.logits, feed_dict={self.x: [[2, 3]]}))
            print(self.sess.run(self.logits, feed_dict={self.x: [[4, 3]]}))


    def evaluate(self,state):
        self.saver.restore(self.sess,save_path='/home/mike/Downloads/reinforelearn/qlearning_nn/')
        a = self.sess.run(self.logits, feed_dict={self.x: state})
        return self.sess.run(tf.argmax(a,1))



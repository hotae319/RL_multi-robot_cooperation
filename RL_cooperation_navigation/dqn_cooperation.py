import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, session, input_size, output_size, name="dqn"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        #self.build_network()
        
    def build_network(self, hidden1_size, hidden2_size, l_rate):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, shape = [None, self.input_size], name = "input_x")
            hid = self.X
            hid = tf.layers.dense(hid, hidden1_size, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
            hid = tf.layers.dense(hid, hidden2_size, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
            hid = tf.layers.dense(hid, self.output_size)
        
            self.Q_pred = hid
            self.Y = tf.placeholder(tf.float32, shape = [None, self.output_size], name = "output_y")
            self.cost = tf.losses.mean_squared_error(self.Y, self.Q_pred)
        
            optimizer = tf.train.AdamOptimizer(learning_rate = l_rate)
            self.train = optimizer.minimize(self.cost)
        
    def predict(self, state):
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self.Q_pred, feed_dict = {self.X: x})
    
    def update(self, x_stack, y_stack):
        feed = {self.X: x_stack, self.Y: y_stack}
        return self.session.run([self.cost,self.train], feed)
          
    
        

import segment_character
import character_process
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_train
def add_layer(inputs, Weights, biases,level_name,activation_function=None):
   # add one more layer and return the output of this layer
   #Weights = tf.Variable(tf.random_normal([in_size, out_size]),dtype=tf.float32,name=level_name+"weights")
   #biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,dtype=tf.float32,name=level_name+"biases")
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs
xs = tf.placeholder(tf.float32, shape=[None, 17])
ys = tf.placeholder(tf.float32, shape=[None, 26])


    # 3.定义神经层：隐藏层和预测层
    # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
w1=tf.Variable(np.arange(17*26).reshape((17,26)),dtype=tf.float32,name='l1weights')
b1=tf.Variable(np.arange(1*26).reshape((1,26)),dtype=tf.float32,name='l1biases')

w2=tf.Variable(np.arange(26*26).reshape((26,26)),dtype=tf.float32,name='predictionweights')
b2=tf.Variable(np.arange(1*26).reshape((1,26)),dtype=tf.float32,name='predictionbiases')

l1 = add_layer(xs, w1, b1, "l1",activation_function=tf.sigmoid)
    #l2 = add_layer(l1, 21, 26, activation_function=tf.sigmoid)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l1, w2, b2,"prediction", activation_function=tf.sigmoid)

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'model2/model.ckpt')
    print(sess.run(w1))
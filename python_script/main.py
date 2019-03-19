import segment_character
import character_process
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_train
import os
import random

animals=["pig","bird","bear","elephant","seal","monkey","tiger"]
def fileName(dir, keyword):
    name = os.listdir(dir)
    length = len(keyword)
    for i in name:
        if len(i) > length and i[0:length] == keyword :
            return i
def similar(a,b):
    result=0
    n=len(a)
    for i in range(n):
        if a[i]==b[i]:
            result+=1
    return result
def animal(animals,sequences):
    ls=len(sequences[0])
    judge=[0,0,0,0,0,0,0]
    for i in range(7):
        if ls==len(animals[i]):
            for s in sequences:
                if similar(s,animals[i])>judge[i]:
                    judge[i]=similar(s,animals[i])
    max=0
    index=-1
    for i in range(7):
        if judge[i]>max:
            index=i
            max=judge[i]
    return index+1

def full_permutation(results):
    n=len(results)
    if n==1:
        return results[0]
    else:
        temp=full_permutation(results[1:])
        result=[]
        for c in results[0]:
            for s in temp:
                result.append(c+s)
        return result

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

addr="Machine Learning AR/Assets/StreamingAssets/111.png"
dir = 'control/'
#print(dir)
inform = fileName(dir[:-1], "inform")
response = fileName(dir[:-1], "response")


##build structure
xs = tf.placeholder(tf.float32, shape=[None, 17])
ys = tf.placeholder(tf.float32, shape=[None, 26])
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
    saver.restore(sess,'model1/model.ckpt')
    #print(sess.run(w1))
    #result=tensorflow_train.max_index(np.array(sess.run(prediction,feed_dict={xs:features})))
    #print(result)
    while True:
        if fileName(r"control","inform") != inform:
            value = fileName(dir[:-1], "value")
            characters = segment_character.segment_character(addr)
            features = []
            for character in characters:
                # plt.imshow(character[0])
                # plt.show()
                # print(character[0])
                features.append(character_process.feature(character_process.frame_cut(character[0])))
            features = (np.array(features)).reshape((-1, 16))
            w, h = np.shape(features)
            features = np.c_[features[:, :], np.ones((w, 1))]
            results = tensorflow_train.sorted_prediction(np.array(sess.run(prediction, feed_dict={xs: features})))
            sequences = full_permutation(results)
            new_value=str(animal(animals,sequences))
            rename = os.rename(dir+value, dir+"value" + new_value + ".txt")
            randomNum = str(random.randint(10000, 99999))
            response = fileName(r"control","response")
            rename = os.rename(dir + response, dir + "response" + randomNum + ".txt")
            inform=fileName(r"control","inform")
            print(new_value)
            print("ok")



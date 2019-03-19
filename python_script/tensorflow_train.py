import tensorflow as tf
import numpy as np
import pandas as pd

# 添加层
def chr_np(y):
    w,h=np.shape(y)
    result=[]
    for i in range(w):
        result.append(chr(int(y[i,0])+ord('a')))
    result=np.array(result)
    result.reshape((-1,1))
    return result
def max_index(prediction):
    w,h=np.shape(prediction)
    result = []
    for i in range(w):
        for j in range(h):
            if prediction[i,j]==np.max(prediction[i]):
                result.append(chr(ord('a')+j))
    result = np.array(result)
    result.reshape((-1, 1))
    return result
def sorted_prediction(prediction):
    w, h = np.shape(prediction)
    sps=[]
    for i in range(w):
        sp = []
        for j in range(h):
            temp=[prediction[i,j],chr(ord('a')+j)]
            sp.append(temp)
        sp=sorted(sp, key=lambda sp:sp[0])[-3::][::-1]
        temp2=[sp[0][1],sp[1][1],sp[2][1]]
        sps.append(temp2)
    sps=np.array(sps)
    return sps
def transform_y(y):
    w=(np.shape(y))[0]
    result=[]
    for i in range(w):
        temp=np.zeros((1,26))
        #print(y[i,0])
        temp[0,int(y[i,0])]=1
        result.append(temp)
    result=(np.array(result)).reshape((-1,26))
    return result
def accurate(p,y):
    w=(np.shape(p))[0]
    i=0.0
    for j in range(w):
        if p[j,]==y[j,]:
            i+=1;
    return i/w
def add_layer(inputs, in_size, out_size,level_name,activation_function=None):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random_normal([in_size, out_size]),dtype=tf.float32,name=level_name+"weights")
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,dtype=tf.float32,name=level_name+"biases")
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs

# 1.训练的数据
# Make up some real data
if __name__=='__main__':
    df=pd.read_csv('csv_train.csv')
    data=np.array(df.values)
    x_data = np.c_[data[:,:-1],np.ones((1300,1))]


    #noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = (data[:,-1]).reshape((-1,1))
    y_data=transform_y(y_data)

    # 2.定义节点准备接收数据
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, shape=[None, 17])
    ys = tf.placeholder(tf.float32, shape=[None, 26])


    # 3.定义神经层：隐藏层和预测层
    # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
    l1 = add_layer(xs, 17, 26, "l1",activation_function=tf.sigmoid)
    #l2 = add_layer(l1, 21, 26, activation_function=tf.sigmoid)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    prediction = add_layer(l1, 26, 26,"prediction", activation_function=tf.sigmoid)

    # 4.定义 loss 表达式
    # the error between prediciton and real data
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                        reduction_indices=[1]))

    # 5.选择 optimizer 使 loss 达到最小
    # 这一行定义了用什么方式去减少 loss，学习率是 0.1
    train_step = tf.train.GradientDescentOptimizer(0.99).minimize(loss)


    # important step 对所有变量进行初始化
    init = tf.initialize_all_variables()
    sess = tf.Session()
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)
    temp_loss=100

    # 迭代 1000 次学习，sess.run optimizer
    k=0
    while(True):
       # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
       k+=1
       sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
       if sess.run(loss, feed_dict={xs: x_data, ys: y_data})>0.2:
           #temp_loss=sess.run(loss, feed_dict={xs: x_data, ys: y_data})
           sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
           # to see the step improvement
           #print(temp_loss)
       else:
           break;
       if k%200==0:
           print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

    saver = tf.train.Saver()
    model_path = 'model2/model.ckpt'
    saver.save(sess,model_path)
    # saver.restore(sess,model_path)
    df=pd.read_csv('csv_test.csv')
    data=np.array(df.values)
    x_data = np.c_[data[:,:-1],np.ones((130,1))]

    #noise = np.random.normal(0, 0.05, x_data.shape)
    #y = tf.get_collection('pred_network')[0]
    y_data = (data[:,-1]).reshape((-1,1))
    result=max_index(np.array(sess.run(prediction,feed_dict={xs:x_data})))
    print(result)
    print('truth')

    print(chr_np(y_data))
    print("accurate rate:",accurate(result,chr_np(y_data))*100,'%')

    print("sorted prediction")
    print(sorted_prediction(sess.run(prediction,feed_dict={xs:x_data})))
# -*- coding: utf-8 -*-
#设计一个函数拟合程序
import tensorflow as tf
import numpy as np


#首先生成一个训练集，主要就是100个点。
#生成x。这个是一个1-100的随机数列
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*x_data + 2*x_data + 3

#print x_data
# print y_data

#开始创建tensorflow结构，A、B、C分别是二次函数的三个参数
#我们设定三个参数的初始值范围，以及维度
A= tf.Variable(tf.random_uniform([1],-0.5,0.5))
B= tf.Variable(tf.random_uniform([1],-3,3))
C= tf.Variable(tf.random_uniform([1],-4,4))

#使用上面定义的参数以及x_data组成带参的二次函数，这个y是就是一个y_data的预测值
y = A*x_data*x_data + B*x_data + C

#计算y与y_data相比的误差，reduce_mean计算出平局值，square计算的是y-ydata的平方，应该
loss = tf.reduce_mean(tf.square(y - y_data))

#创建一个优化器，GradientDescentOptimizer拥有一种非常基础的优化策略，梯度下降算法，形参应该是类似于预测步长之类的参数
#我觉得如果这里的学习效率越低应该需要的训练次数就越多，但是训练的就会越精准。如果这个值太大就会出问题的
opt = tf.train.GradientDescentOptimizer(0.1)

#创建一个训练器，使用梯度下降算法减少y_data与y之间的误差
trainer = opt.minimize(loss)

#在神经网络中初始化变量
init = tf.initialize_all_variables()

#这里tensorflow结构就建立完了

#创建一个会话，应该就是一个任务的意思
sess = tf.Session();

#使用初始化器激活这个训练任务
sess.run(init)

#从这里开始训练
for stepNum in range(100000):
    sess.run(trainer)#执行一次训练
    #每10部打印出来一个训练的结果
    if stepNum % 10 == 0:
        #run这个函数的意义应该就是在神经网络中执行一些东西，执行变量就会返回变量名
        print "stepNum =", stepNum ,sess.run(A),sess.run(B),sess.run(C)



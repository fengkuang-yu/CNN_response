# -*- coding: utf-8 -*-
"""
Created on Mon May 14 18:46:52 2018

@author: yyh
"""

import scipy.io as sio  
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

# Set random seed
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

Batch=50
REGULARIZATION_RATE=0.0001
Datasize=20
TestDataSize=288
TimeIntervals=8
Iteration=5000
LEARNING_RATE=1e-4
PredIntervals=0   #0 denotes 5mins
  
# =============================================================================
# 读入训练和测试数据
data=pd.read_excel('DATA/input_data.xlsx',
                   Sheetname='Sheet1',header=None)
labels=pd.read_excel('DATA/input_labels.xlsx',
                   Sheetname='Sheet1',header=None)
test_data=pd.read_excel('DATA/test_data_10days.xlsx',
                   Sheetname='Sheet1',header=None)
test_labels=pd.read_excel('DATA/test_data_labels_10days.xlsx',
                   Sheetname='Sheet1',header=None)
# =============================================================================

# =============================================================================
# 将数据处理为所需类型
input_data=data.as_matrix()
Imagesize=Datasize*TimeIntervals
ColumnNum=len(input_data)
SampleNum=ColumnNum-TimeIntervals

#生成训练样本
input_data1=np.zeros((SampleNum,Imagesize))
for i in range(SampleNum):
    temp=input_data[i:i+TimeIntervals,:]
    input_data1[i,:]=temp.reshape(1,Imagesize)

#输入训练样本labels
input_data2=labels.as_matrix()

#输入10天的测试数据
test_data_10days=test_data.as_matrix()
TestSampleNum=len(test_data_10days)-TimeIntervals-PredIntervals
input_data3=np.zeros((TestSampleNum,Imagesize))
for i in range(TestSampleNum):
    temp=test_data_10days[i:i+TimeIntervals,:]
    input_data3[i,:]=temp.reshape(1,Imagesize)

#输入10天的测试数据的标签
input_data4=test_labels.as_matrix()
# =============================================================================

x = tf.placeholder("float", shape=[None, Imagesize])
y_= tf.placeholder("float", shape=[None, 1])

sess = tf.InteractiveSession()

###################函数定义###########################
#初始化权矩阵
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#初始化偏置矩阵
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#卷积操作函数
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#2*2最大化池化操作
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

###############主程序开始#########################
  
#第一层使用64个卷积核
W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1,Datasize,TimeIntervals,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积层使用32个卷积核
W_conv2 = weight_variable([3, 3, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第一层全连接层
W_fc1 = weight_variable([Imagesize*2, 16]) #Datasize/4*5*32
b_fc1 = bias_variable([16])
h_pool2_flat = tf.reshape(h_pool2, [-1, Imagesize*2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout的概率（这里没使用dropout,因为效果不是很好）
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#第二层全连接
W_fc2 = weight_variable([16, 1])
b_fc2 = bias_variable([1])
y_fc2=tf.matmul(h_fc1_drop, W_fc2) + b_fc2


MAPE_error =tf.reduce_sum(tf.abs(y_fc2-y_)/y_)/Batch

regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
regularization = regularizer(W_fc1)+regularizer(W_fc2)
loss=MAPE_error+regularization
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
sess.run(tf.global_variables_initializer())

for i in range(Iteration):
    Sample1=np.random.randint(1,12000,size=(1,Batch))
    train_datas=input_data1[Sample1].reshape(Batch,Imagesize)
    Sample2=Sample1+TimeIntervals+PredIntervals
    train_label=input_data2[Sample2].reshape(Batch,1)
    if i%100 == 0:
        a = MAPE_error.eval(feed_dict={x:train_datas,
                                 y_: train_label, keep_prob: 1.0})
        print ("step %d, training loss is %g"%(i, a))
    train_step.run(feed_dict={x: train_datas, y_: train_label,
                              keep_prob: 1})

Pred=np.zeros((1,TestSampleNum))
for i in range(TestSampleNum):
   test_datas=input_data3[i:i+1,:]
   test_labels=input_data4[i+TimeIntervals+PredIntervals:i+TimeIntervals+PredIntervals+1,:]
   e = y_fc2.eval(feed_dict={x:test_datas, y_: test_labels, keep_prob: 1.0})
   Pred[0,i] = e[0,0]
   train_step.run(feed_dict={x:test_datas, y_: test_labels, keep_prob: 1.0})


# =============================================================================
# 画图开始
#1.选择出画图的一天，[2304:2592,:]表示的是画图的那一天，Python选取不到2592
#2.Predictin是预测的画图当天的结果，减TimeIntervals的原因是ten_test_label是从0开
#  始到len(test_data)，而Prediction是从0开始到len(test_data-TimeIntervals)
#
real_data=input_data4[2304:2592,:] 
Prediction=Pred.reshape(TestSampleNum,-1)
Prediction1=Prediction[2304-TimeIntervals-PredIntervals:2592-TimeIntervals-PredIntervals,:]
AbsoluteE1=real_data-Prediction1
d=abs(AbsoluteE1)

sio.savemat('prediction_CNN_filtered',{'matrix2':Prediction1})  # 写入mat文件
##第二幅图
## =============================================================================
print('For CNN datasize is %d, time intervals is %d, prediction time is %d'%(Datasize,TimeIntervals,5+PredIntervals*5))
# =============================================================================
# 计算一天的MAPE和MAE值
sum_=0
for i in range(TestDataSize):
    sum_=sum_+(d[i]/real_data[i])
mape=(1/(TestDataSize))*(sum_)
print('One day MAPE=', mape)

sum_=0;
for i in range(TestDataSize):
    sum_=sum_+d[i]
mae=sum_/(TestDataSize)
print('One day MAE=', mae)

# =============================================================================

# =============================================================================
# 计算十天的MAPE和MAE
real_data=input_data4[TimeIntervals+PredIntervals:,:]
#Prediction=Pred.reshape(TestSampleNum,-1)
AbsoluteE2=real_data-Prediction
d=abs(AbsoluteE2)
sum_=0
for i in range(TestSampleNum):
    sum_=sum_+(d[i]/real_data[i])
mape=(1/(TestSampleNum))*(sum_)
print('Ten days MAPE=', mape)


sum_=0
for i in range(TestSampleNum):
    sum_=sum_+d[i]
mae=sum_/(TestSampleNum)
print('Ten days MAE=', mae)

# =============================================================================

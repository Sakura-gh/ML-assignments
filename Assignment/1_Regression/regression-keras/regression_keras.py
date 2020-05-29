#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np

data=pd.read_csv('../data/train.csv',encoding='big5')
data


# In[2]:


# 我们只需要第三列以后的data
data = data.iloc[:,3:]
# 将NR的数据全部置为0
data[data == 'NR'] = 0
# 将dataframe幻化成numpy
raw_data = data.to_numpy()

#查看raw_data
raw_data


# In[3]:


# 现在要开始处理Training data
# 由于提供的数据是每个月前20天(每天24小时)的资料(共18种类型)，因此每个月的这20*24=480个小时的时间是连续的，每10个小时可以作为一笔data，480小时共471笔data
# 由于不同月份之间的时间是不连续的，因此先把所有的资料按照月份分开
month_data = {}
for month in range(12):
  # 每个月共20*24=480份数据，共18种空气成分，每一种成分每月都有480份数据，因此初始化一个18*480的array
  temp_data = np.empty([18,480])
  for day in range(20):
    # temp_data中加入第day天的数据，由于每天都有24份数据共24列,故列的范围是24*day~24*(day+1)；选择加入temp_data的是第20×month+day这一天的数据，由于每天有18种大气成分的数据共18行，因此行的范围是18*(20*month+day)~18*(20*month+day+1)
    temp_data[:, 24 * day : 24 * (day + 1)] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1)]
  # 把这个月的data放进month_data[month]里去
  month_data[month]=temp_data

# 查看第一个月的数据，并借此查看数据的结构：每一行都是一种大气成分在24*20个连续小时内的值
month_data[0]


# In[4]:


# 这个时候我们就要在12段连续的时间里每10个小时取出一笔Training data
# 每个月连续时间有20*24=480h，每10h作为一笔data，共可以分为471笔data；一共12个月，因此共12*471笔data
# 每一笔data，前9h为input，最后1h的PM2.5为output，因此input是一个18*9的矩阵，如果把它摊平就是一个18*9的feature vector

# x是input，共12*471笔data，每个input是一个18*9的feature vector
x = np.empty([12 * 471, 18 * 9], dtype = float)
# y是output，共12*471笔data，每个output是第10个小时的第9种大气成分PM2.5，因此只有1维
y = np.empty([12 * 471, 1], dtype = float)

# 取出input和output
for month in range(12):
  # 每个月份共471笔data，依次遍历即可
  for i in range(471):
    # 用一个18行(18种物质),10列(10个小时的数据，包括input和output)的temp_data来存放每一笔Training data
    temp_data = np.empty([18, 10])
    # 第i笔data的范围是i~i+9，共10列，注意这里i+10是开区间，实际只取到了i+9
    temp_data = month_data[month][:, i : i + 10]
    # 将temp_data前9列的作为input的数据(18行9列)用reshape平摊到一个18*9的行向量上
    x[471 * month + i, :] = temp_data[:, 0 : 9].reshape(1,-1)
    # temp_data的第10列作为output，只有第10行的PM2.5值是有用的
    y[471 * month + i] = temp_data[9, 9]

print(x)
print(y)


# In[5]:


# 这个时候我们已经有了Training data的x和y，但由于feature的每一个dimension大小范围都是不一样的，因此还要对其做feature scale
# 求出期望mean和标准差std，利用公式x'=(x-u)/σ来使同一列的数据同时满足同一个分布

# 直接调用numpy的mean和std计算期望和标准差，axis=0表示对列进行计算
mean_x = np.mean(x, axis = 0)
std_x = np.std(x, axis = 0)

# 对每一个feature都进行normalize归一化处理
for i in range(len(x)):
  for j in range(len(x[0])):
    if std_x[j] != 0:
      x[i][j] = (x[i][j] - mean_x[j])/std_x[j]

# 查看normalize后的input x
x


# In[6]:


# 为了有效衡量testing data的bias对结果的影响，这里切出一块validaiton set
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation_set = x[math.floor(len(x) * 0.8): , :]
y_validation_set = y[math.floor(len(y) * 0.8): , :]

print('x_train_set len: ' + str(len(x_train_set)) + '\n')
print(x_train_set)
print('---------------------------------------------------------------------------')
print('y_train_set len: ' + str(len(y_train_set)) + '\n')
print(y_train_set)
print('---------------------------------------------------------------------------')
print('x_validation_set len: ' + str(len(x_validation_set)) + '\n')
print(x_validation_set)
print('---------------------------------------------------------------------------')
print('y_validation_set len: ' + str(len(y_validation_set)) + '\n')
print(y_validation_set)
print('---------------------------------------------------------------------------')


# In[7]:


# 处理数据,主要是testing data
# 此时的x不需要再用np.concatenate函数来拼接一个常数项的feature，no need:x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

testdata = pd.read_csv('../data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
# no need: test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x


# In[8]:


# 用Keras搭建Regression的神经网络
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.layers import Dropout

x_train = x_train_set
y_train = y_train_set
# DNN 100 5
model = Sequential()
model.add(Dense(input_dim = len(x_train[0]), units = 50, activation = 'tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size = 100, epochs = 200)
result_train = model.evaluate(x_train, y_train)
result_train

rmse_train = np.sqrt(np.sum(np.power(model.predict(x_train)-y_train, 2)) / (len(y_train)))
print('training_set rmse: %6f'%rmse_train)
predict_y_validation_set = model.predict(x_validation_set)
rmse = np.sqrt(np.sum(np.power(predict_y_validation_set - y_validation_set, 2)) / (len(y_validation_set)))
print('validation_set rmse: %6f'%rmse)


# In[9]:


# # 保存model
# model.save('regression_keras.h5')


# In[10]:


# # 载入保存的model，测试得到的结果与之前训练的是否相同
# from keras.models import load_model
# model1=load_model('regression_keras.h5')

# rmse_train = np.sqrt(np.sum(np.power(model1.predict(x_train)-y_train, 2)) / (len(y_train)))
# print('training_set rmse: %6f'%rmse_train)
# predict_y_validation_set = model1.predict(x_validation_set)
# rmse = np.sqrt(np.sum(np.power(predict_y_validation_set - y_validation_set, 2)) / (len(y_validation_set)))
# print('validation_set rmse: %6f'%rmse)


# In[11]:


# 用全部的data对model进行训练
x_train = x
y_train = y
# DNN 
model = Sequential()
model.add(Dense(input_dim = len(x_train[0]), units = 50, activation = 'tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size = 100, epochs = 200)
result_train = model.evaluate(x_train, y_train)
result_train

rmse_train = np.sqrt(np.sum(np.power(model.predict(x_train)-y_train, 2)) / (len(y_train)))
print('training_all rmse: %6f'%rmse_train)


# In[12]:


# 保存用全部data训练好的model
model.save('regression_keras_final.h5')


# In[13]:


# 加载model并预测
from keras.models import load_model

model_all_data = load_model('regression_keras_final.h5')
rmse_train = np.sqrt(np.sum(np.power(model_all_data.predict(x_train)-y_train, 2)) / (len(y_train)))
print('training_all rmse: %6f'%rmse_train)

predict_y_keras = model_all_data.predict(test_x)
predict_y_keras


# In[14]:


import csv
with open('predict_keras_all_data.csv', mode='w', newline='') as predict_file:
    csv_writer = csv.writer(predict_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), predict_y_keras[i][0]]
        csv_writer.writerow(row)
        print(row)


# In[ ]:





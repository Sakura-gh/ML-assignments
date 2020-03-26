#!/usr/bin/env python
# coding: utf-8

# In[63]:


import sys
import pandas as pd
import numpy as np

# 读取事先存放好的Training data
data=pd.read_csv('../data/train.csv',encoding='big5')
data


# In[64]:


# 我们只需要第三列以后的data
data = data.iloc[:,3:]
# 将NR的数据全部置为0
data[data == 'NR'] = 0
# 将dataframe幻化成numpy
raw_data = data.to_numpy()

#查看raw_data
raw_data


# In[65]:


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


# In[66]:


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


# In[67]:


# 这个时候我们已经有了Training data的x和y，但由于feature的每一个dimension大小范围都是不一样的，因此还要对其做feature scale
# 求出期望mean和标准差std，利用公式x'=(x-u)/σ来使同一列的数据同时满足同一个分布

# 由于要尝试不同normalize的效果，这里先用x_new保留一份原始的x,先用copy()函数复制一份x
x_new = x.copy()

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


# In[68]:


# 求mean和std的另一种思路，由于18*9个feature中，每9个feature都是属于同一种大气成分，因此可以9个一组进行求mean和std，而不是1个feature求一个mean和std
# 因此总共有18种物质，即18种mean和std

# 先初始化为0
mean_x_new = np.zeros(18)
std_x_new = np.zeros(18)
# 每9列分为一组求mean和std,共18组
for i in range(18):
    mean_x_new[i] = np.mean(x_new[:, 9*i : 9*(i+1)])
    std_x_new[i] = np.std(x_new[:, 9*i : 9*(i+1)])


# 对feature进行归一化处理
for i in range(18):
    if std_x_new[i] != 0:
        x_new[:, 9*i : 9*(i+1)] = (x_new[:, 9*i : 9*(i+1)] - mean_x_new[i]) / std_x_new[i]

# 查看使用新的normalize后的x_new
x_new


# In[69]:


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


# In[39]:


# 现在开始尝试简单的linear Regression: y = w_1*x_1+w_2*x_2+...+b
# 由于常数项的存在，dimension需要在18*9的基础上再加上一维，初始化每一维对应的weight为0，并进行gradient descent，经过多次iteration来学习
import matplotlib.pyplot as plt

dim = 18 * 9 + 1
w = np.zeros([dim, 1])
# 利用np.concatenate函数在原先的x上加上一个常数的feature=1,用一个全1的列向量，并使用axis=1来横向拼接
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

# 设定初始化的learning rate
learning_rate = 0.1
# 设定迭代次数iteration
iter_time = 100000
# adagrad表达式为w'=w-lr*gd/sqrt(Σgd^2))，为了避免分母为0，加上一个很小的eps
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
root_mean_square = np.empty([iter_time, 1])

# loss function = (x*w-y)^2, loss对w求导为 2*x*(x*w-y),即gradient
for i in range(iter_time):
  # 计算loss、gradient和Adagrad
  loss = np.sum(np.power(np.dot(x, w) - y, 2)) 
  gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
  adagrad += gradient ** 2 # 跟np.power()的效果一样
  # 更新w,注意这里gradient、adagrad都是列向量，加减乘除的用法跟常数一样
  w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
  # 计算root mean square
  root_mean_square[i][0] = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / (471 * 12))
  print('root_mean_square_%-6d: '%i + str(root_mean_square[i][0]))

# save w
np.save('weight_adagrad.npy', w)
w

plt.plot(range(len(root_mean_square)),root_mean_square)
plt.show()


# In[70]:


# 用新的normalize的new_x进行同样的训练，观察训练效果
dim = 18 * 9 + 1
w_new = np.zeros([dim, 1])
# 利用np.concatenate函数在原先的x上加上一个常数的feature=1,用一个全1的列向量，并使用axis=1来横向拼接
x_new = np.concatenate((np.ones([12 * 471, 1]), x_new), axis = 1).astype(float)
#x_new.shape
# 设定初始化的learning rate
learning_rate = 0.1
# 设定迭代次数iteration
iter_time = 100000
# adagrad表达式为w'=w-lr*gd/sqrt(Σgd^2))，为了避免分母为0，加上一个很小的eps
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
root_mean_square = np.empty([iter_time, 1])

# loss function = (x*w-y)^2, loss对w求导为 2*x*(x*w-y),即gradient
for i in range(iter_time):
  # 计算loss、gradient和Adagrad
  loss = np.sum(np.power(np.dot(x_new, w_new) - y, 2)) 
  gradient = 2 * np.dot(x_new.transpose(), np.dot(x_new, w_new) - y)
  adagrad += gradient ** 2 # 跟np.power()的效果一样
  # 更新w,注意这里gradient、adagrad都是列向量，加减乘除的用法跟常数一样
  w_new = w_new - learning_rate * gradient / np.sqrt(adagrad + eps)
  # 计算root mean square
  root_mean_square[i][0] = np.sqrt(np.sum(np.power(np.dot(x_new, w_new) - y, 2)) / (471 * 12))
  print('root_mean_square_%-6d: '%i + str(root_mean_square[i][0]))

# save w
np.save('weight_adagrad_new.npy', w_new)
w_new

plt.plot(range(len(root_mean_square)),root_mean_square)
plt.show()


# In[71]:


testdata = pd.read_csv('../data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i : 18 * (i + 1), :].reshape(1,-1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x


# In[75]:


w = np.load('weight_adagrad.npy')
predict_y = np.dot(test_x, w)
predict_y


# In[76]:


w_new = np.load('weight_adagrad_new.npy')
predict_y_new = np.dot(test_x, w_new)
predict_y_new


# In[77]:


import csv
with open('predict_adagrad.csv', mode = 'w', newline = '') as predict_file:
    csv_writer = csv.writer(predict_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), predict_y[i][0]]
        csv_writer.writerow(row)
        print(row)
        


# In[79]:


with open('predict_adagrad_new.csv', mode = 'w', newline = '') as predict_new_file:
    csv_writer = csv.writer(predict_new_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), predict_y_new[i][0]]
        csv_writer.writerow(row)
        print(row)


# In[ ]:





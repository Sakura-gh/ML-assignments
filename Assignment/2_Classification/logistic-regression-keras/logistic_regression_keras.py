#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 下载资料并做normalize，切为Training set和validation set
import numpy as np

np.random.seed(0)

x_train_fpath = '../data/X_train'
y_train_fpath = '../data/Y_train'
x_test_fpath  = '../data/X_test'

# 第一行是feature的名称，所以先执行next(f)跳过第一行的内容；第一个dimension是id，feature[1:]从第二个dimension开始读取
with open(x_train_fpath) as f:
    next(f)
    x_train = np.array([line.strip('\n').split(',')[1:]  for line in f], dtype = float)

with open(y_train_fpath) as f:
    next(f)
    y_train = np.array([line.strip('\n').split(',')[1]  for line in f], dtype = float)
    
with open(x_test_fpath) as f:
    next(f)
    x_test = np.array([line.strip('\n').split(',')[1:]   for line in f], dtype = float)
    
print('x_train :\n',x_train,x_train.shape,'\n')
print('y_train :\n',y_train,y_train.shape,'\n')
print('x_test :\n',x_test,x_test.shape)


# In[2]:


def _normalize(x, train = True, specified_column = None, x_mean = None, x_std = None):
    '''
    This function normalizes specific columns of x
    注意，testing data要跟training data的normalize方式一致，要用training data的mean和std，
    因此还需要input已知的x_mean和x_std
    '''
    # 如果没有指定列，那就穷举所有列，这里np.arange类似于range函数，只不过前者创造的对象是array类型
    if specified_column == None:
        specified_column = np.arange(x.shape[1])
    
    # train=True: for training data; train=False: for testing data，只计算training data的mean和std
    if train:
        # axis=0，对指定列求mean，注意np.mean返回的是一个列向量，因此需要用reshape(1,-1)转化成行向量
        x_mean = np.mean(x[:, specified_column], axis = 0).reshape(1, -1)
        # axis=0，对指定列求std
        x_std  = np.std(x[:, specified_column], axis = 0).reshape(1, -1)
     
    # 对指定列进行normalize，注意相减的两个向量行数不同但列数相同，相当于前者的每一行都减去x_mean这个行向量，除法同理
    # 分母加一个很小很小的数是为了避免标准差为0
    x[:, specified_column] = (x[:, specified_column] - x_mean) / (x_std + 1e-8)
    
    return x, x_mean, x_std


# In[3]:


def _train_split(x, y, validation_ratio = 0.25):
    '''
    This function splits data into training set and validation set
    '''
    train_size = int(len(x) * (1 - validation_ratio))
    
    #return x,y of training set and validation set  
    # 如果返回值为x[:train_size, :]的话会报错，但这两种形式本质上是一样的，存疑
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]


# In[4]:


# normalize training data and testing data
x_train, x_mean, x_std = _normalize(x_train, train = True)
x_test, _, _ = _normalize(x_test, train = False, x_mean = x_mean, x_std = x_std)

# split training data into training set and validation set
x_training_set, y_training_set, x_validation_set, y_validation_set = _train_split(x_train, y_train, validation_ratio = 0.1)

print('x_training_set : ', x_training_set.shape, '\n', x_training_set)
print('------------------------------------------------------------------------')
print('y_training_set : ', y_training_set.shape, '\n', y_training_set)
print('------------------------------------------------------------------------')
print('x_validation_set : ', x_validation_set.shape, '\n', x_validation_set)
print('------------------------------------------------------------------------')
print('y_validation_set : ', y_validation_set.shape, '\n', y_validation_set)


# In[5]:


from keras.utils import to_categorical

# 要将y转化成one-hot编码，才能给keras训练
y_training_set = to_categorical(y_training_set)
y_validation_set = to_categorical(y_validation_set)
print(y_training_set, '\n\n', y_validation_set)


# In[6]:


# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras import regularizers

# 刚开始训练的时候在Training set上acc高达96%，但在validation set上的performance并没有那么好，为了减少overfitting，这里用了regularization和dropout，虽然Training set的准确率下降了，但validation set的准确率却有一定程度的提高
model = Sequential()
model.add(Dense(input_dim = len(x_training_set[0]), units = 50, activation = 'relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dropout(0.5))
# model.add(Dense(units = 250, activation = 'relu')) # activity_regularizer=regularizers.l1(0.0001)
# model.add(Dropout(0.5))
model.add(Dense(units = 50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 2, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adagrad', metrics = ['accuracy'])
model.fit(x_training_set, y_training_set, batch_size = 100, epochs = 20)

predict_training_set = model.evaluate(x_training_set, y_training_set)
predict_validation_set = model.evaluate(x_validation_set, y_validation_set)

print('Training set Acc :', predict_training_set[1])
print('Validation set Acc :', predict_validation_set[1])


# In[7]:


# 把所有的y转化为one-hot编码
y_train = to_categorical(y_train, 2)


# In[8]:


# 利用validation set挑选好model的参数以后，用所有的data对model进行再次训练
model = Sequential()
model.add(Dense(input_dim = len(x_train[0]), units = 50, activation = 'relu', kernel_regularizer = regularizers.l2(0.0001)))
model.add(Dropout(0.5))
model.add(Dense(units = 50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 2, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adagrad', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 100, epochs = 20)


# In[9]:


y_test_predict = np.round(model.predict(x_test)[:, 1]).astype(int)
y_test_predict


# In[10]:


import csv
with open('predict_keras.csv', mode = 'w', newline = '') as f:
    csv_writer = csv.writer(f)
    header = ['id', 'label']
    print(header)
    csv_writer.writerow(header)
    for i in range(y_test_predict.shape[0]):
        row = [str(i), y_test_predict[i]]
        print(row)
        csv_writer.writerow(row)


# In[ ]:





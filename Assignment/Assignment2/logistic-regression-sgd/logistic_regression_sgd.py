#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

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


# some useful functions
# np.dot()的作用主要体现在两个1-D向量相乘和一个N-D矩阵和一个1-D向量相乘的情景下：
# 两个1-D向量A与B相乘(A、B元素数量必须相等)：等价于A、B对应元素相乘并累计求和，最终得到一个常量积；注意A*B和np.dot(A,B)的区别，前者是A、B对应元素相乘，每次相乘的积都作为新的1-D向量的一个元素，而不是把这些积累加为一个常量
# 一个N-D矩阵A和一个1-D向量B相乘(A的每一行元素数量必须与B的元素数量相等)：等价于把这个N-D矩阵A拆成N个1-D向量，它们分别与B做1-D矩阵的相乘，得到的积作为结果的一个元素，总共有N个积，最终的结果就是由这N个积组成的1-D向量
# np.dot()在w*x上的应用可以减少转置，在其他方面也有比较便利的应用
def _shuffle(x, y):
    '''
    This function shuffles two equal-length list/array, x and y, together
    '''
    # 打乱原本的次序
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    
    return x[randomize], y[randomize]

def _sigmoid(z):
    '''
    sigmoid function can be used to calculate probability
    To avoid overflow, minimum/maximum output value is set
    '''
    # np.clip(a, a_min, a_max)将数组a限制在a_min和a_max之间，超出范围的值将被赋以边界值
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1 - (1e-6))

def _f(x, w, b):
    '''
    logistic regression function, parameterized by w and b
    
    Arguements:
        X: input data, shape = [batch_size, data_dimension]
        w: weight vector, shape = [data_dimension, ]
        b: bias, scalar
    output:
        predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    '''
    # np.dot特别适合用来计算x*w，无需转置，直接就是N维x的每一行与一维w相乘得到的结果汇总成一个一维的y
    return _sigmoid(np.dot(x, w) + b)

def _predict(x, w, b):
    '''
    This function returns a truth value prediction for each row of x
    by round function to make 0 or 1
    '''
    # 利用round函数的四舍五入功能把概率转化成0或1
    return np.round(_f(x, w, b)).astype(np.int)
    
def _accuracy(y_predict, y_label):
    '''
    This function calculates prediction accuracy
    '''
    # 预测值和标签值相减，取绝对值后再求平均，相当于预测错误的个数(差为1)/总个数，即错误率，1-错误率即正确率
    acc = 1 - np.mean(np.abs(y_predict - y_label))
    
    return acc


# In[6]:


# loss function(cross_entropy) and gradient
def _cross_entropy_loss(y_predict, y_label):
    '''
    This function computes the cross entropy 
    
    Arguements:
        y_pred: probabilistic predictions, float vector
        Y_label: ground truth labels, bool vector
    Output:
        cross entropy, scalar    
    '''
    # cross_entropy = -Σ(y_head*ln(y)+(1-y_head)*ln(1-y))
    # 注意，这里的np.dot可以直接计算两个一位矩阵的积(前提是元素个数一致)，无需转置；np.log()等价于ln()
    # 因此这里的矩阵积实际上已经做了Σ的工作，左边的np.dot计算的是Σ(y_head*ln(y)，右边的np.dot计算的是Σ(1-y_head)*ln(1-y))
    cross_entropy = -(np.dot(y_label, np.log(y_predict)) + np.dot((1 - y_label), np.log(1 - y_predict)))
    
    return cross_entropy

def _gradient(x, y_label, w, b):
    '''
    This function computes the gradient of cross entropy loss with respect to weight w and bias b
    loss function: -Σ (y_head*ln(y)+(1-y_head)*ln(1-y)), 分别对w和b求偏微分，可得
    gradient of w: -Σ（y_head - y)*x
    gradient of b: -Σ（y_head - y)
    '''
    y_predict = _f(x, w, b)
    # 也可以是w_gradient = -np.dot(x.T, y_label - y_predict)
    w_gradient = -np.sum((y_label - y_predict) * x.T, 1)
    b_gradient = -np.sum(y_label - y_predict)
    
    return w_gradient, b_gradient


# In[7]:


# mini-batch stochastic-gradient-descent

train_size = x_training_set.shape[0]
validation_size = x_validation_set.shape[0]
dim = x_training_set.shape[1]
# initialize w and b
w = np.zeros(dim)
b= np.zeros(1)

# parameters for training
max_iter = 100
batch_size = 10
learning_rate = 1

# save the loss and accuracy
training_set_loss = []
training_set_acc = []
validation_set_loss = []
validation_set_acc = []

w_adagrad = 1e-8
b_adagrad = 1e-8

# calcuate the number of parameter updates
step = 1

# training for iterations
for epoch in range(max_iter):
    # random shuffle at the beginning of each epoch
    x_training_set, y_training_set = _shuffle(x_training_set, y_training_set)
    
    # mini-batch training
    for i in range(int(np.floor(train_size / batch_size))):
        # get the mini-batch
        x = x_training_set[i * batch_size : (i + 1) * batch_size]
        y = y_training_set[i * batch_size : (i + 1) * batch_size]
        
        # compute the gradient
        w_gradient, b_gradient = _gradient(x, y, w, b)
        
        # compute the adagrad
        w_adagrad = w_adagrad + np.power(w_gradient, 2)
        b_adagrad = b_adagrad + np.power(b_gradient, 2)
        
        # gradient descent update 
        # learning rate decay with time
        w = w - learning_rate * w_gradient / np.sqrt(w_adagrad)
        b = b - learning_rate * b_gradient / np.sqrt(b_adagrad)
        
        step = step + 1
        
    # one epoch: compute loss and accuracy of training set and validation set
    y_training_predict = _predict(x_training_set, w, b) # predict函数将Probability取round，只剩下0和1
    y_probability = _f(x_training_set, w, b) # Probability用来计算cross_entropy，不能用round后的值，否则会出现ln(0)的错误
    acc = _accuracy(y_training_predict, y_training_set)
    loss = _cross_entropy_loss(y_probability, y_training_set) / train_size # average cross_entropy
    training_set_acc.append(acc)
    training_set_loss.append(loss)
    print('training_set_acc_%d   : %f \t training_set_loss_%d  : %f'%(epoch, acc, epoch, loss))
    
    y_validation_predict = _predict(x_validation_set, w, b)
    y_probability = _f(x_validation_set, w, b)
    acc = _accuracy(y_validation_predict, y_validation_set)
    loss = _cross_entropy_loss(y_probability, y_validation_set) / validation_size # average cross_entropy
    validation_set_acc.append(acc)
    validation_set_loss.append(loss)
    
# validation_set的acc和loss只输出最后那一次
print('validation_set_acc_%d : %f \t validation_set_loss_%d : %f'%(epoch, acc, epoch, loss))
print()
   


# In[8]:


import matplotlib.pyplot as plt
# loss curve
plt.plot(training_set_loss)
plt.plot(validation_set_loss)
plt.title('Loss')
plt.legend(['training_set', 'validation_set'])
plt.savefig('loss_sgd.png')
plt.show()

# accuracy curve
plt.plot(training_set_acc)
plt.plot(validation_set_acc)
plt.title('Accuracy')
plt.legend(['training_set', 'validation_set'])
plt.savefig('acc_sgd.png')
plt.show()


# In[9]:


# mini-batch stochastic-gradient-descent

train_size = x_train.shape[0]
dim = x_train.shape[1]
# initialize w and b
w = np.zeros(dim)
b= np.zeros(1)

# parameters for training
max_iter = 1000
batch_size = 10
learning_rate = 1

# save the loss and accuracy
train_loss = []
train_acc = []

w_adagrad = 1e-8
b_adagrad = 1e-8


# training for iterations
for epoch in range(max_iter):
    # random shuffle at the beginning of each epoch
    x_train, y_train = _shuffle(x_train, y_train)
    
    # mini-batch training
    for i in range(int(np.floor(train_size / batch_size))):
        # get the mini-batch
        x = x_train[i * batch_size : (i + 1) * batch_size]
        y = y_train[i * batch_size : (i + 1) * batch_size]
        
        # compute the gradient
        w_gradient, b_gradient = _gradient(x, y, w, b)
        
        # compute the adagrad
        w_adagrad = w_adagrad + np.power(w_gradient, 2)
        b_adagrad = b_adagrad + np.power(b_gradient, 2)
        
        # gradient descent update 
        # learning rate decay with time
        w = w - learning_rate * w_gradient / np.sqrt(w_adagrad)
        b = b - learning_rate * b_gradient / np.sqrt(b_adagrad)
    
    # one epoch: compute loss and accuracy of training set and validation set
    y_train_predict = _predict(x_train, w, b) # predict函数将Probability取round，只剩下0和1
    y_probability = _f(x_train, w, b) # Probability用来计算cross_entropy，不能用round后的值，否则会出现ln(0)的错误
    acc = _accuracy(y_train_predict, y_train)
    loss = _cross_entropy_loss(y_probability, y_train) / train_size # average cross_entropy
    train_acc.append(acc)
    train_loss.append(loss)
    print('train_acc_%d   : %f \t train_loss_%d  : %f'%(epoch, acc, epoch, loss))
        


# In[10]:


# loss curve
plt.plot(train_loss)
plt.title('Loss')
plt.legend(['train'])
plt.show()

# accuracy curve
plt.plot(train_acc)
plt.title('Accuracy')
plt.legend(['train'])
plt.show()

np.save('weight_adagrad_sgd.npy', w)
np.save('bias_adagrad_sgd.npy', b)


# In[11]:


# predict testing data
import csv
w = np.load('weight_adagrad_sgd.npy')
b = np.load('bias_adagrad_sgd.npy')
y_test_predict = _predict(x_test, w, b)
print(y_test_predict, y_test_predict.shape)

with open('predict_adagrad_sgd.csv', mode = 'w', newline = '') as f:
    csv_writer = csv.writer(f)
    header = ['id', 'label']
    print(header)
    csv_writer.writerow(header)
    for i in range(y_test_predict.shape[0]):
        row = [str(i), y_test_predict[i]]
        csv_writer.writerow(row)
        print(row)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

x_train_fpath = '../data/X_train'
y_train_fpath = '../data/Y_train'
x_test_fpath  = '../data/X_test'

with open(x_train_fpath, mode = 'r') as f:
    next(f)
    x_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

with open(y_train_fpath, mode = 'r') as f:
    next(f)
    y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
    
with open(x_test_fpath, mode = 'r') as f:
    next(f)
    x_test  = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    
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


# normalize training data and testing data
x_train, x_mean, x_std = _normalize(x_train, train = True)
x_test, _, _ = _normalize(x_test, train = False, x_mean = x_mean, x_std = x_std)


# In[4]:


# zip的作用是把x和y打包成一个元组，然后根据y的值就能够挑选出所有y=0的x和y=1的x
x_train_0 = np.array([x for x, y in zip(x_train, y_train) if y == 0])
x_train_1 = np.array([x for x, y in zip(x_train, y_train) if y == 1])

# 对两组x重新求mean
mean_0 = np.mean(x_train_0, axis = 0)
mean_1 = np.mean(x_train_1, axis = 0)

# 计算feature的维数
data_dim = x_train.shape[1]

# 计算in-class covariance
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in x_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / x_train_0.shape[0]
for x in x_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / x_train_1.shape[0]
    
# 对in-class covariance进行加权平均，并用来共享
cov = (cov_0 * x_train_0.shape[0] + cov_1 * x_train_1.shape[0]) / (x_train_0.shape[0] + x_train_1.shape[0])

cov


# In[5]:


def _sigmoid(z):
    '''
    sigmoid function can be used to calculate probability
    To avoid overflow, minimum/maximum output value is set
    '''
    # np.clip(a, a_min, a_max)将数组a限制在a_min和a_max之间，超出范围的值将被赋以边界值
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

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
    return _sigmoid(np.matmul(x, w) + b)

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


# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices = False)
inv = np.dot(v.T * 1 / s, u.T)

# Directly compute weights and bias
w = np.dot(inv, mean_0 - mean_1)
b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))    + np.log(float(x_train_0.shape[0]) / x_train_1.shape[0]) 

# Compute accuracy on training set
y_train_predict = 1 - _predict(x_train, w, b)
print('Training accuracy: {}'.format(_accuracy(y_train_predict, y_train)))


# In[7]:


import csv
y_test_predict = 1 - _predict(x_test, w, b)
with open('predict_generative_model.csv', mode = 'w', newline = '') as f:
    csv_writer = csv.writer(f)
    header = ['id', 'label']
    print(header)
    csv_writer.writerow(header)
    for i in range(y_test_predict.shape[0]):
        row = [str(i), y_test_predict[i]]
        print(row)
        csv_writer.writerow(row)


# In[ ]:





# Assignment1

在kaggle上的结果(评判标准是rmse)：

- simple_baseline = 8.73773
- strong_baseline = 7.14231
- private score为7.49105对应的排名为257/423
- keras搭建神经网络造成的overfitting比较明显，手写linear model的gradient descent的performance会比较好(data不是很多的情况下)
- 可以做进一步的特征工程

|         method         | Public Score | Private Score |
| :--------------------: | :----------: | :-----------: |
|  predict_adagrad_new   |   5.45971    |    7.49105    |
|    predict_adagrad     |   5.45813    |    7.50907    |
|      predict_adam      |   5.61442    |    7.67992    |
| predict_keras_all_data |   6.23082    |   11.32828    |
|     predict_keras      |   6.26881    |    7.93896    |


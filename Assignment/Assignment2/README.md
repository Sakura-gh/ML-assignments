# Assignment2

在kaggle上的结果(评判标准是Categorization Accuracy)：

- simple_baseline = 0.88675
- strong_baseline = 0.89102

- private score为0.89110对应的排名是99/285
- 手写linear的cross entropy的gradient descent的performance往往会比用keras写神经网络要来的好
- 特征工程是进一步优化的思路

|       method        | Public Score | Private Score |
| :-----------------: | :----------: | :-----------: |
| predict_adagrad_gd  |   0.88878    |    0.89110    |
| predict_adagrad_gd  |   0.88929    |    0.89052    |
| predict_adagrad_sgd |   0.88842    |    0.88863    |
|    predict_keras    |   0.88769    |    0.88900    |
|    predict_keras    |   0.88798    |    0.88675    |


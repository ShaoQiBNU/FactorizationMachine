# Factorization Machine 算法详解

## 背景
> 在计算广告和推荐系统中，CTR预估(click-through rate)是非常重要的一个环节，判断一个物品是否进行推荐需要根据CTR预估的点击率排序决定。业界常用的方法有人工特征工程 + LR（Logistic Regression）、GBDT（Gradient Boosting Decision Tree） + LR、FM（Factorization Machine）和FFM（Field-aware Factorization Machine）模型。本文主要介绍FM算法的原理和应用实例。

## 原理

### FM的提出目的

> 与SVM对比来看，FM算法采用了factorized parametrization建模优化，主要是解决稀疏数据下的特征组合问题，并且其预测的复杂度是线性的，对于连续和离散特征有较好的通用性。在高稀疏的数据集下，SVM的表现不尽如意，而FM算法则可以非常好的解决这一问题。

### 数据

> 以电影评分数据集为例，数据集有用户和电影的特征，需要预测用户对没有看过的电影的评分。如下所示：

img1

> 评分是label，用户id、电影id、评分时间是特征。由于用户id和电影id是categorical类型的，需要经过独热编码（One-Hot Encoding）转换成数值型特征。因为是categorical特征，所以经过one-hot编码以后，导致样本数据变得很稀疏。
>
> 每行表示目标 ![[公式]](https://www.zhihu.com/equation?tex=y%5E%7B%28i%29%7D) 与其对应的特征向量 ![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%28i%29%7D) ，蓝色区域表示了用户变量，红色区域表示了电影变量，黄色区域表示了其他隐含的变量，进行了归一化，绿色区域表示一个月内的投票时间，棕色区域表示了用户上一个评分的电影，最右边的区域是评分。

### FM算法建模

普通的线性模型例如LR，是将各个特征独立考虑的，并没有考虑到特征与特征之间的相互关系。但实际上，特征之间可能具有一定的关联。以新闻推荐为例，一般男性用户看军事新闻多，而女性用户喜欢情感类新闻，可以看出性别与新闻的频道有一定的关联性，如果能找出这类的特征，对提升推荐系统的效果是非常有意义的。为了简单起见，只考虑二阶交叉的情况，具体模型如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D%28x%29%3Dw_%7B0%7D%2B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_%7Bi%7Dx_%7Bi%7D%7D%2B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Csum_%7Bj%3Di%2B1%7D%5E%7Bn%7D%7Bw_%7Bij%7Dx_%7Bi%7Dx_%7Bj%7D%7D%7D+%5Ctag%7B1%7D+%5C%5C+)

其中， ![[公式]](https://www.zhihu.com/equation?tex=n) 代表样本的特征数量， ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 是第i个特征的值， ![[公式]](https://www.zhihu.com/equation?tex=w_%7B0+%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bi%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bij+%7D) 是模型参数，只有当 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 与 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bj%7D) 都不为0时，交叉才有意义。

在数据稀疏的情况下，满足交叉项不为0的样本将非常少，当训练样本不足时，很容易导致参数 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bij%7D) 训练不充分而不准确，最终影响模型的效果。
那么，交叉项参数的训练问题可以用矩阵分解来近似解决，有下面的公式。

## 代码实例



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
> 每行表示目标<a href="https://www.codecogs.com/eqnedit.php?latex=y^{(i)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y^{(i)}" title="y^{(i)}" /></a>与其对应的特征向量<a href="https://www.codecogs.com/eqnedit.php?latex=x^{(i)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x^{(i)}" title="x^{(i)}" /></a> ，蓝色区域表示了用户变量，红色区域表示了电影变量，黄色区域表示了其他隐含的变量，进行了归一化，绿色区域表示一个月内的投票时间，棕色区域表示了用户上一个评分的电影，最右边的区域是评分。

### FM算法建模

> 普通的线性模型例如LR，是将各个特征独立考虑的，并没有考虑到特征与特征之间的相互关系。但实际上，特征之间可能具有一定的关联。以新闻推荐为例，一般男性用户看军事新闻多，而女性用户喜欢情感类新闻，可以看出性别与新闻的频道有一定的关联性，如果能找出这类的特征，对提升推荐系统的效果是非常有意义的。为了简单起见，只考虑二阶交叉的情况，具体模型如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y(x)}&space;=&space;\omega&space;_{0}&space;&plus;&space;\sum_{i=1}^{n}\omega&space;_{i}&space;*x&space;_{i}&space;&plus;&space;\sum_{i=1}^{n}&space;\sum_{j=i&plus;1}^{n}\omega&space;_{i,j}&space;*x&space;_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\tilde{y(x)}&space;=&space;\omega&space;_{0}&space;&plus;&space;\sum_{i=1}^{n}\omega&space;_{i}&space;*x&space;_{i}&space;&plus;&space;\sum_{i=1}^{n}&space;\sum_{j=i&plus;1}^{n}\omega&space;_{i,j}&space;x&space;_{i}x_{j}" title="\tilde{y(x)} = \omega _{0} + \sum_{i=1}^{n}\omega _{i} *x _{i} + \sum_{i=1}^{n} \sum_{j=i+1}^{n}\omega _{i,j}x _{i}x_{j}" /></a>

> 其中， <a href="https://www.codecogs.com/eqnedit.php?latex=n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n" title="n" /></a> 代表样本的特征数量， <a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{i}" title="x_{i}" /></a> 是第i个特征的值， ![[公式]](https://www.zhihu.com/equation?tex=w_%7B0+%7D) 、 <a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{i}" title="\omega _{i}" /></a> 、 <a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{i,j}" title="\omega _{i,j}" /></a>是模型参数，只有当  <a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{i}" title="x_{i}" /></a>  与  <a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{j}" title="x_{j}" /></a> 都不为0时，交叉才有意义。在数据稀疏的情况下，满足交叉项不为0的样本将非常少，当训练样本不足时，很容易导致参数  <a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{i,j}" title="\omega _{i,j}" /></a> 训练不充分而不准确，最终影响模型的效果。FM算法采用矩阵分解来近似解决交叉项参数的训练问题，公式如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y(x)}&space;=&space;\omega&space;_{0}&space;&plus;&space;\sum_{i=1}^{n}\omega&space;_{i}&space;*x&space;_{i}&space;&plus;&space;\sum_{i=1}^{n}&space;\sum_{j=i&plus;1}^{n}<v&space;_{i},&space;v&space;_{j}>&space;x&space;_{i}x_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\tilde{y(x)}&space;=&space;\omega&space;_{0}&space;&plus;&space;\sum_{i=1}^{n}\omega&space;_{i}&space;*x&space;_{i}&space;&plus;&space;\sum_{i=1}^{n}&space;\sum_{j=i&plus;1}^{n}<v&space;_{i},&space;v&space;_{j}>&space;x&space;_{i}x_{j}" title="\tilde{y(x)} = \omega _{0} + \sum_{i=1}^{n}\omega _{i} *x _{i} + \sum_{i=1}^{n} \sum_{j=i+1}^{n}<v _{i}, v _{j}> x _{i}x_{j}" /></a>

> 其中，<a href="https://www.codecogs.com/eqnedit.php?latex=<v_{i},&space;v_{j}>" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<v_{i},&space;v_{j}>" title="<v_{i}, v_{j}>" /></a>表示





## 代码实例



# Factorization Machine 算法详解

## 背景
> 在计算广告和推荐系统中，CTR预估(click-through rate)是非常重要的一个环节，判断一个物品是否进行推荐需要根据CTR预估的点击率排序决定。业界常用的方法有人工特征工程 + LR（Logistic Regression）、GBDT（Gradient Boosting Decision Tree） + LR、FM（Factorization Machine）和FFM（Field-aware Factorization Machine）模型。本文主要介绍[FM算法](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)的原理和应用实例。

## 原理

### FM的提出目的

> 与SVM对比来看，FM算法采用了factorized parametrization建模优化，主要是解决稀疏数据下的特征组合问题，并且其预测的复杂度是线性的，对于连续和离散特征有较好的通用性。在高稀疏的数据集下，SVM的表现不尽如意，而FM算法则可以非常好的解决这一问题。

### 数据

> 以电影评分数据集为例，数据集有用户和电影的特征，需要预测用户对没有看过的电影的评分。如下所示：

![image](https://github.com/ShaoQiBNU/FactorizationMachine/blob/master/img/1.jpg)

> 评分是label，用户id、电影id、评分时间是特征。由于用户id和电影id是categorical类型的，需要经过独热编码（One-Hot Encoding）转换成数值型特征。因为是categorical特征，所以经过one-hot编码以后，导致样本数据变得很稀疏。
>
> 每行表示目标<a href="https://www.codecogs.com/eqnedit.php?latex=y^{(i)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y^{(i)}" title="y^{(i)}" /></a>与其对应的特征向量<a href="https://www.codecogs.com/eqnedit.php?latex=x^{(i)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x^{(i)}" title="x^{(i)}" /></a> ，蓝色区域表示了用户变量，红色区域表示了电影变量，黄色区域表示了其他隐含的变量，进行了归一化，绿色区域表示一个月内的投票时间，棕色区域表示了用户上一个评分的电影，最右边的区域是评分。

### FM算法建模

> 普通的线性模型例如LR，是将各个特征独立考虑的，并没有考虑到特征与特征之间的相互关系。但实际上，特征之间可能具有一定的关联。以新闻推荐为例，一般男性用户看军事新闻多，而女性用户喜欢情感类新闻，可以看出性别与新闻的频道有一定的关联性，如果能找出这类的特征，对提升推荐系统的效果是非常有意义的。为了简单起见，只考虑二阶交叉的情况，具体模型如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y(x)}&space;=&space;\omega&space;_{0}&space;&plus;&space;\sum_{i=1}^{n}\omega&space;_{i}&space;*x&space;_{i}&space;&plus;&space;\sum_{i=1}^{n}&space;\sum_{j=i&plus;1}^{n}\omega&space;_{i,j}&space;*x&space;_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\tilde{y(x)}&space;=&space;\omega&space;_{0}&space;&plus;&space;\sum_{i=1}^{n}\omega&space;_{i}&space;*x&space;_{i}&space;&plus;&space;\sum_{i=1}^{n}&space;\sum_{j=i&plus;1}^{n}\omega&space;_{i,j}&space;x&space;_{i}x_{j}" title="\tilde{y(x)} = \omega _{0} + \sum_{i=1}^{n}\omega _{i} *x _{i} + \sum_{i=1}^{n} \sum_{j=i+1}^{n}\omega _{i,j}x _{i}x_{j}" /></a>

> 其中， <a href="https://www.codecogs.com/eqnedit.php?latex=n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n" title="n" /></a> 代表样本的特征数量， <a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{i}" title="x_{i}" /></a> 是第i个特征的值， ![[公式]](https://www.zhihu.com/equation?tex=w_%7B0+%7D) 、 <a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{i}" title="\omega _{i}" /></a> 、 <a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{i,j}" title="\omega _{i,j}" /></a>是模型参数，只有当  <a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{i}" title="x_{i}" /></a>  与  <a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{j}" title="x_{j}" /></a> 都不为0时，交叉才有意义。在数据稀疏的情况下，满足交叉项不为0的样本将非常少，当训练样本不足时，很容易导致参数  <a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{i,j}" title="\omega _{i,j}" /></a> 训练不充分而不准确，最终影响模型的效果。
>
> FM算法采用矩阵分解来近似解决交叉项参数的训练问题，公式如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y(x)}&space;=&space;\omega&space;_{0}&space;&plus;&space;\sum_{i=1}^{n}\omega&space;_{i}&space;*x&space;_{i}&space;&plus;&space;\sum_{i=1}^{n}&space;\sum_{j=i&plus;1}^{n}<v&space;_{i},&space;v&space;_{j}>&space;x&space;_{i}x_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\tilde{y(x)}&space;=&space;\omega&space;_{0}&space;&plus;&space;\sum_{i=1}^{n}\omega&space;_{i}&space;*x&space;_{i}&space;&plus;&space;\sum_{i=1}^{n}&space;\sum_{j=i&plus;1}^{n}<v&space;_{i},&space;v&space;_{j}>&space;x&space;_{i}x_{j}" title="\tilde{y(x)} = \omega _{0} + \sum_{i=1}^{n}\omega _{i} *x _{i} + \sum_{i=1}^{n} \sum_{j=i+1}^{n}<v _{i}, v _{j}> x _{i}x_{j}" /></a>

> 模型优化的参数为：<a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{0}\in&space;R,&space;\omega\in&space;R^{n},&space;V\in&space;R^{n\times&space;k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{0}\in&space;R,&space;\omega\in&space;R^{n},&space;V\in&space;R^{n\times&space;k}" title="\omega _{0}\in R, \omega\in R^{n}, V\in R^{n\times k}" /></a>，<a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{0}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{0}" title="\omega _{0}" /></a>是global bias，<a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{i}" title="\omega _{i}" /></a>是<a href="https://www.codecogs.com/eqnedit.php?latex=x&space;_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x&space;_{i}" title="x _{i}" /></a>的权重，<a href="https://www.codecogs.com/eqnedit.php?latex=<v_{i},&space;v_{j}>" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<v_{i},&space;v_{j}>" title="<v_{i}, v_{j}>" /></a>则建模了<a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{i}" title="x_{i}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=x_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{j}" title="x_{j}" /></a>的组合关系。其中，<a href="https://www.codecogs.com/eqnedit.php?latex=<v_{i},&space;v_{j}>" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<v_{i},&space;v_{j}>" title="<v_{i}, v_{j}>" /></a>表示向量的内积，向量embedding的维度设置为k（超参数，具体可以调整），如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=<v_{i},&space;v_{j}>&space;=&space;\sum_{f=1}^{k}v_{i,f}\cdot&space;v_{j,f}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<v_{i},&space;v_{j}>&space;=&space;\sum_{f=1}^{k}v_{i,f}\cdot&space;v_{j,f}" title="<v_{i}, v_{j}> = \sum_{f=1}^{k}v_{i,f}\cdot v_{j,f}" /></a>

> <a href="https://www.codecogs.com/eqnedit.php?latex=V" target="_blank"><img src="https://latex.codecogs.com/svg.latex?V" title="V" /></a>中每一行向量<a href="https://www.codecogs.com/eqnedit.php?latex=v_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?v_{i}" title="v_{i}" /></a>代表<a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_{i}" title="x_{i}" /></a>的embedding，即 k factors。对任意正定矩阵 <a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/svg.latex?W" title="W" /></a>，只要k足够大，就存在矩阵 <a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/svg.latex?W" title="W" /></a> ，使得<a href="https://www.codecogs.com/eqnedit.php?latex=W&space;=&space;V&space;V_{T}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?W&space;=&space;V&space;V_{T}" title="W = V V_{T}" /></a>。然而在数据稀疏的情况下，应该选择较小的k，因为没有足够的数据来估计<a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/svg.latex?W" title="W" /></a> 。限制k的大小提高了模型更好的泛化能力，以上述电影评分系统为例，假设要计算用户 A 与电影 ST 的交叉项，然而训练数据里没有这种情况，这样会导致<a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;_{A,&space;ST}=0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\omega&space;_{A,&space;ST}=0" title="\omega _{A, ST}=0" /></a> ，但是我们可以近似计算出<a href="https://www.codecogs.com/eqnedit.php?latex=<V_{A},&space;V_{ST}>" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<V_{A},&space;V_{ST}>" title="<V_{A}, V_{ST}>" /></a> 。首先，用户 B 和 C 有相似的向量<a href="https://www.codecogs.com/eqnedit.php?latex=V_{B}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?V_{B}" title="V_{B}" /></a> 和 <a href="https://www.codecogs.com/eqnedit.php?latex=V_{C}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?V_{C}" title="V_{C}" /></a>，因为他们对 SW的预测评分比较相似， 所以<a href="https://www.codecogs.com/eqnedit.php?latex=<V_{B},&space;V_{SW}>" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<V_{B},&space;V_{SW}>" title="<V_{B}, V_{SW}>" /></a> 和 <a href="https://www.codecogs.com/eqnedit.php?latex=<V_{C},&space;V_{SW}>" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<V_{C},&space;V_{SW}>" title="<V_{C}, V_{SW}>" /></a>是相似的。用户 A 和 C 有不同的向量，因为对 TI 和 SW 的预测评分完全不同。接下来， ST 和 SW 的向量可能相似，因为用户 B 对这两部电影的评分也相似。最后可以看出， <a href="https://www.codecogs.com/eqnedit.php?latex=<V_{A},&space;V_{ST}>" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<V_{A},&space;V_{ST}>" title="<V_{A}, V_{ST}>" /></a> 与 <a href="https://www.codecogs.com/eqnedit.php?latex=<V_{A},&space;V_{SW}>" target="_blank"><img src="https://latex.codecogs.com/svg.latex?<V_{A},&space;V_{SW}>" title="<V_{A}, V_{SW}>" /></a> 是相似的。

> 直接计算FM的二阶交叉项复杂度较大，可以通过公式变换，减少到线性复杂度，如下：


![image](https://github.com/ShaoQiBNU/FactorizationMachine/blob/master/img/2.jpg)


### 应用

> FM算法可以应用到很多预测任务中，如下：
>
> - regression：采用最小平方误差优化<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y(x)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\tilde{y(x)}" title="\tilde{y(x)}" /></a>
> - binary classification：使用hinge或logit误差函数来优化 the sign of <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y(x)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\tilde{y(x)}" title="\tilde{y(x)}" /></a>
> - ranking：向量<a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x" title="x" /></a> 通过 <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y(x)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\tilde{y(x)}" title="\tilde{y(x)}" /></a>的分数排序，并且通过pairwise的分类损失来优化成对的样本<a href="https://www.codecogs.com/eqnedit.php?latex=(x^{a},&space;x^{b})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(x^{a},&space;x^{b})" title="(x^{a}, x^{b})" /></a>

### 参数学习

> 模型的参数可以通过梯度下降的方法（例如随机梯度下降）来学习，对于各种的损失函数。FM模型的梯度是：

![image](https://github.com/ShaoQiBNU/FactorizationMachine/blob/master/img/3.jpg)

> 由于 <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{j=1}^{n}v_{j,f}x_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sum_{j=1}^{n}v_{j,f}x_{j}" title="\sum_{j=1}^{n}v_{j,f}x_{j}" /></a> 只与 f 有关，与 i 是独立的，可以提前计算出来，并且每次梯度更新可以在常数时间复杂度内完成，因此FM参数训练的复杂度也是O(kn)。综上可知，FM可以在线性时间训练和预测，是一种非常高效的模型。

### 高阶FM

> 2阶FM很容易泛化到高阶，如下：

![image](https://github.com/ShaoQiBNU/FactorizationMachine/blob/master/img/4.jpg)


## FM算法与SVM、其他Factorization models对比

> 论文还从原理和结果上对比了FM算法与SVM、其他Factorization models的性能和特点，具体可参考论文。

## FM算法的优势

1. 在高度稀疏的情况下特征之间的交叉仍然能够估计，而且可以泛化到未被观察的交叉
2. 参数的学习和模型的预测的时间复杂度是线性的

## 代码实例

具体应用可以参考：

https://github.com/rixwew/pytorch-fm

https://github.com/A1fcc/FM-FFM

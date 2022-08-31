# MatchNet无法训练问题解决方案

## 问题描述

在训练过程中，参照与原论文[^1]相同的数据域处理方式及训练超参数，得到的Loss值不收敛，具体表现为Loss值始终维持在0.69～0.71附近。

## 解决思路

由于torch和mindspore不同的默认初始化方式（本模型中出现差异的结构为[2D卷积层](https://www.mindspore.cn/docs/zh-CN/r1.8/note/api_mapping/pytorch_diff/nn_Conv2d.html)和[全连接层](https://www.mindspore.cn/docs/zh-CN/r1.8/note/api_mapping/pytorch_diff/Dense.html)），考虑是否由于参数初始化导致。调整为相同的参数初始化方式后，训练Loss反而变为NaN，遂进一步排查。

考虑到本文所使用的损失函数为交叉熵损失，且对于
$$
-ln(\frac{1}{2})\approx0.693
$$
第一反应想到网络经过Softmax输出logits时结果大约为`[0.5,0.5]`，即网络初始化时就已经收敛到某局部最优点，导致无效学习，而调整初始化参数却又导致Loss变为NaN，结合该模型全连接结构中并无Dropout操作，猜测是否训练过程中容易出现梯度爆炸现象。故暂提出方案：

1）调整参数初始化方式为2D卷积层：``mindspore.common.initializer.Orthogonal(gain=0.6)`(能使得非线性网络结构得到一个与深度无关的训练时间优化[^2])

全连接层：`mindspore.common.initializer.XavierUniform(gain=1)`（可以使得输入值x的方差和经过网络层后的输出值y的方差一致[^3]）

2）在1）的基础上，适当减小learning rate：0.01 —> 0.001

最终成功完成训练。

---

## 参考文献

>   [^1]:Han, Xufeng, Thomas Leung, Yangqing Jia, Rahul Sukthankar和Alexander C Berg. 《Matchnet: Unifying feature and metric learning for patch-based matching》, 3279–86, 2015.
>   [^2]:Saxe, Andrew M., James L. McClelland和Surya Ganguli. 《Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks》. arXiv, 2014年2月19日. http://arxiv.org/abs/1312.6120.
>   [^3]:Understanding the difficulty of training deep feedforward neural networksGlorot, Xavier, 和Yoshua Bengio. 《Understanding the Difﬁculty of Training Deep Feedforward Neural Networks》, 不详, 8.

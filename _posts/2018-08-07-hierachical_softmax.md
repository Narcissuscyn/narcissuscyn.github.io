---
layout: post
title: "hierachical softmax"
date: 2018-08-07 17:40:41
categories: 机器学习
tags: 损失函数
mathjax: true
---

* content
{:toc}

要理解这个算法，就要先知道word2vector里面发两个模型：CBOW和skip-gram模型。要理解其中理论，此文讲解甚好：[（1）Word2Vector中的数学理解](https://spaces.ac.cn/usr/uploads/2017/04/2833204610.pdf)、[（2）Deep Learning 实战之 word2vec](https://spaces.ac.cn/usr/uploads/2017/04/146269300.pdf)。

#### 我的一些理解

##### **语言模型（及词向量）**


	词向量就是将语言数学化的表示，以能够输入算法，进行学习预测等。而词向量用于`统计语言模型`中就是一个重要的基础；统计语言模型是所有nlp的基础，它被广泛应用于语音识别、机器翻译等各项任务；统计语言模型就是计算`一个句子的概率`的模型：$p(W)$
$$
p(W)=	p(w_1^T)=p(w_1,...,w_T)
$$
贝叶斯公式在统计机器学习中有着重要作用，利用贝叶斯模型，可将统计语言模型改写为：
$$
p(W)=p(w_1)p(w_2|w_1)p(w_3|w_12)...p(w_T|w_1{T-1})
$$
给出条件概率了，模型处理起来的方法就多了，计算模型参数的模型常有n-gram、神经网络、决策树、最大熵、最大熵马尔科夫、条件随机场等。（1）中主要讲解了n-gram模型和神经网络模型。

1. 对于n-gram模型，我的理解是：它认为一个词出现的概率只和它前面的n个词相关，即n-1阶的Markov假设，即：
   $$
   p(w_k|w_1^{k-1})=p(w_k|w_{k-n+1}^{k-1})
   $$
   根据组合词的出现词频统计可以计算得到$p(w_k|w_1^{k-1})$。例如n=2时：
   $$
   p(w_k|w_1^{k-1})\approx \frac{count(w_{k-1},w_k)}{count(w_{k-1})}
   $$
   参数与n的关系：

   | n                 | 参数数量 |
   | ----------------- | -------- |
   | 2                 | 4*10^10  |
   | 1                 | 2*10^5   |
   | `3`(实际中最常用) | 8*10^15  |
   | 4                 | 16*10^20 |

2. 对于神经网络模型，我的理解是首先建立一个目标函数$\prod_{w\in C}p(w|content(w))$而content(w)就是与w这个单词相关的上下文单词，在n-gram模型中就是$w_{i-n+1}^{i-1}$。利用最大对数似然就可以得到：
   $$
   L=\sum_{w \in C}log(p(w|content(w)))
   $$
   然后对似然函数进行最大化,这里都和n-gram模型十分类似，都通过建立的条件概率来进行计算语言模型，但不同的是，神经网络模型对$p(w|content(w))$的建模不同，不是通过词频统计来建模的，而是通过神经网络模型来建模的：$p(w|content(w))=F(w,context(w),\theta)$而神经网络模型中就用到了单词的`词向量`（在n-gram模型中是不需要的,因为它只对单词做词频统计，然后保存这个概率即可，实际上来说，它也不存在真正意义上的参数，有点类似于knn，主要在于存储部分）。而说到词向量，就有两种方式：1）one-hot repersentation;2)distributed representation.区别可见参考文献（1）。

   ![](https://ws1.sinaimg.cn/large/005IsqTWly1fu1h6x7rggj30lp0aigne.jpg)

   这里面`v(w)`,W,U,p,q就是参数,注意v(w)（即单词w的词向量，而不是输入context的词向量哦！）也要经过训练才能得到。最后$y_w$经过一个softmax归一化之后，就可以得到最终概率：
   $$
   p(w|content(w))=\frac{e^{y_w,i_w}}{\sum_{i=1}^{N}e^{y_w,i}}
   $$
   

##### **CBOW 模型和 skip gram模型**

这两个模型主要是用于生成distributed wordvector的。

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu1hj9itv4j30m30damyx.jpg)



我觉得这两个模型并没有特别大的区别，都类似于神经网络的模型，唯一的区别就是对条件概率的建模方式不同，CBOW是建立$p(w|content(w))$,而后者则是建立$p(content(w)|w)$的模型，两个模型的具体结构参考（1）



##### **hierachiccal softmax和negtive sampling**

	CBOW和ship-gram都可以分别和这两种多分类方法相结合使用，negtive sampling不再使用hierachiccal softmax中的`huffman树`，而是利用相对简单的`随机负采样`。hierachiccal softmax原始思想是层级的分类:

> 对于二叉 树来说，则是使用二分类近似原来的多分类。例如给定 $w_i$，先让模型判断 $w_o$是 不是名词，再判断是不是食物名，再判断是不是水果，再判断是不是“桔子”。 虽然 word2vec 论文里，作者是使用哈夫曼编码构造的一连串两分类。但是在训 练过程中，模型会赋予这些抽象的中间结点一个合适的向量， 这个向量代表了它 对应的所有子结点。因为真正的单词公用了这些抽象结点的向量，所以 Hierarchical Softmax 方法和原始问题并不是等价的，但是这种近似并不会显著带 来性能上的损失同时又使得模型的求解规模显著上升    

在这个地方采用huffman树来实现这个，仿佛并不能完全的体现多层级、父类包含子类的思想。



**举个例子**：CBOW+hierachical softmax(看成多个二分类):

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu1ijdylosj30fn0ekwh8.jpg)
$$
p(“足球”|context("足球"))=\prod_{j=2}^{5}p(d_j^w|X_w,\theta_{j-1}^w)
$$
对于模型的的损失函数推导和求导 详细见(1)
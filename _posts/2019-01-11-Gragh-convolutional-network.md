---
layout: post
title:  "graph convolutional network思想简介"
date:   2019-01-11 17:40:41
categories: 图像理解
tags: frameworks
mathjax: true
author: Narcissus
---

* content
{:toc}


注：https://zhuanlan.zhihu.com/p/37091549，这篇blog写的很好，我借鉴其关键内容并重新整理了一下，添加了一些自己的看法。

## 为什么有图卷积网络

卷积神经网络的**缺点**：它研究的对象还是限制在Euclidean domains的数据。什么是**euclidean data**？ Euclidean data**最显著的特征就是有规则的空间结构**，比如图片是规则的正方形栅格，比如语音是规则的一维序列。而**这些数据结构能够用一维、二维的矩阵表示**，卷积神经网络处理起来很高效。



但是，我们的现实生活中有很多数据并不具备规则的空间结构，称为**non-Euclidean data**。如图像中的关系网络，检测出的物体为节点，节点之间的关系为边。这些图谱结构每个节点连接都不尽相同，有的节点有三个连接，有的节点有两个连接，是不规则的数据结构。



## 图的基本特征

- 每个节点都有自己的特征信息
- 每个节点具有结构信息
  - 就是说具有节点之间的连接信息

## 图卷积网络

### 定义

一种能**自动化**（不需要手动设计特征）地同时学到图结构数据的**特征信息**与**结构信息**的方法；

### 目标

目标是提取出这种广义图结构的特征，进而完成一些任务，如标签补全等。比如在图像中的**关系结构图**，标签往往都不是穷尽的，那么是否可以从这个思路上面去做一些多标签的研究呢？

### 模型框架

![img](https://pic2.zhimg.com/80/v2-26a2515386ee987aee3e8d1c3d9032e9_hd.jpg)

### 图卷积算子

![img](https://pic3.zhimg.com/80/v2-57873af06a427e88512bb5348857395e_hd.jpg)

![img](https://pic1.zhimg.com/80/v2-4a26ad90044bbe05541292c36f8d02c0_hd.jpg)

### 计算流程

- 发射：每一个节点将自身的特征信息经过变换后发送给邻居节点
- 接收：每个节点将邻居节点的特征信息聚集起来，是在对**节点的局部结构信息**进行融合，这里信息的多少和感受野的大小有关
  - 这里说明我PANet那篇文章对谓词建模的有效性，因为所有谓词之间都存在负向的或者正向的关系，那同时利用到所有谓词的特征都是关键的，而图卷积网络需要根据感受野的大小来利用有效的节点信息，这样就大打折扣了；
  - 另外，这里有一个改进方案。attention的机制，或许在这里可以和感受野等结合起来，让谓词网络利用对当前谓词最有用的其他相关谓词。
  - 变换：把前面的信息聚集之后做非线性变换，增加模型的表达能力。

### 特征

#### 具有和卷积神经网络一样的特点

- 局部参数共享；图卷积算子适用于每个节点，处处共享
- receptive field正比于层数，层数越多，感受域就更广，参与运算的信息就更多

#### 具备深度学习的三个性质

- 层级结构（特征一层一层抽取，一层比一层更抽象，更高级）
- 非线性变换 （增加模型的表达能力）
- 端对端训练（不需要再去定义任何规则，只需要给图的节点一个标记，让模型自己学习，融合特征信息和结构信息。）

#### 本身具有的四个特征

- GCN 是对卷积神经网络在 **graph domain** 上的自然**推广**
- 能同时对节点特征信息与结构信息进行端对端学习，是目前对图数据学习任务的最佳选择。
- 图卷积适用性极广，适用于任意拓扑结构的节点与图。
- **节点分类**与**边预测**等任务上，在公开数据集上效果要远远优于其他方法。



## 参考文献

https://zhuanlan.zhihu.com/p/37091549

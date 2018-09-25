---
layout: post
title:  "relation network for object detection"
date:   2018-09-25
categories: RelationshipDetection ObjectDetect
tags: ObjectDetect relationship
mathjax: true
author: Narcissus
---

* content
{:toc}
# 整体思路

![图1 网络结构](https://ws1.sinaimg.cn/large/005IsqTWgy1fvm0sxgcjkj30fb0aq41o.jpg)

<center>图1 整体结构</center>

主要是在原来的四阶段检测的基础上增加了**关系信息模块**，在**instance recognition**模块，FC模块和relation模块的输入输出都是一样的，因此能直接加入到FC层之后，这个relation模块主要是利用了每个物体与其他物体之间的关系，它的建立是基于原始attention model改的。而**duplicate removal**模块主要是代替NMS，进行一个二分类，去除一些重复的框。



![图2](https://ws1.sinaimg.cn/large/005IsqTWgy1fvm4usktx5j30gl0ditat.jpg)

<center>图2 两个relation模块</center>

# Object Relation module

## 原始attention model

主要思想是使用Attention机制，最原始的attention模型[56]：
$$
v^{out}=softmax(\frac{qK^t}{\sqrt{d_k}})V
$$
这个模型叫`Scaled Dot-product Attention`,q为query，K为keys，$d_k$是q的维度，V是value；具体指什么，为什么这么建模是，要去看看引用论文。

## 本文使用attention model

1. 本文主要是对每个物体n,根据其他的m个物体建立以下关系特征：
   $$
   f_R(n)=\sum_m{w^{mn}·(W_V·f_A^m)}
   $$

   - $W_V$是线性变换矩阵，相当于公式(1)中的V；
   - $f_A^m$是第m个物体的visual feature；
   - $w^{mn}$是第m个物体对第n个物体的权重，其实这里就体现了attention的思想，把注意力更多的给哪些物体；

2. $w^{mn}$的计算：

   它的计算来自$w^{mn}_G$和$w_A^{mn}$,分别表示第m个物体对第n个物体的几何特征(geometry feature)权重和视觉特征(visual feature)权重;
   $$
   w^{mn}=\frac{W_G^{mn}exp(w_A^{mn})}{\sum_k{W_G^{kn}exp(w_A^{kn})}}
   $$

3. $w_A^{mn}$的计算：

   这部分类似于公式(1)的attention model：
   $$
   w_A^{mn}=\frac{dot({W_Kf_A^m},{W_Qf^n_A})}{\sqrt{d_k}}
   $$
   其实为什么会这么建模，我也并不是十分的清除啊！$W_K$,$W_Q$类似于公式(1)中的K和q。

4. $w_G^{mn}$的计算：
   $$
   w_G^{mn}=max\{0,W_G·\epsilon_G(f_G^m,f_G^n)\}
   $$

   - 其中，$f_G$是$(log(\frac{\lvert{x_m}-{x_n}\rvert}{w_m},log(\frac{\lvert{y_m}-{y_n}\rvert}{h_m},log(\frac{w_n}{w_m}),log(\frac{h_n}{h_m}))$,是论文[23]的改进版本，log操作主要是考虑距离较远的物体，而原始的box regression只考虑近处的物体。

   - $\epsilon_G$是为了将geometry feature 投影到高维空间；

   - $W_G$是为了将特征变换成向量；

   - max操作是进行一个trim操作

     > The zero trimming operation restricts relations only between objects of certain geometric relationships

5. 最终特征：
   $$
   f_A^n=f_A^n+Concat(f_R^1(n),···,f_R^{N_r}(n)),for\ all\ n
   $$
   这里值得主要的是采用的concate操作，而并没使用add操作;

图2中这个(a)2fc模块 就是加入了object relation之后的结构:

![](https://ws1.sinaimg.cn/large/005IsqTWgy1fvm52lk5lpj30e104g74y.jpg)

# duplicate relation module

这个模块就没有用attention机制了，就是利用score、visual feature、bbox构建了一个二分类器，即图2中(b)：

- $f^n$(1024-d feature，应该是整张图像的visual feature) 和$score^n$得到visual feature；
- 结合bbox,得到所有物体的appearance feature；
- 类别分数是排序的顺序，而不是原始的分数a rank $\in[1, N] $




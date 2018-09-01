---
layout: post
title: "visual_relationship_detection_with_object_spatial_distribution"
date: 2018-08-06 17:40:41
mathjax: true
---


## 《visual_relationship_detection_with_object_spatial_distribution》

* content
{:toc}
### 文章思想

	文章主要是`language proir[12]`那篇文章上进行修改的，主要是增加了`spatial distribution`,它包含了position relation、size relation、shape relation以及distance relation,从而构成一个region model部分。然后把这个模块加入到C(*)目标函数，以及正则化项里面。其实并没有太大的贡献，实验结果也并不是很好。虽然文章中说到一些对数据的统计结果，但实际上，只是针对大部分的，我想对于特殊情况的关系，这先验统计知识是具有副作用的。



### 为什么设计这样的spatial distribution？

原因是这个region model的设计经过以下统计：

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu104p7p01j30te09rq4m.jpg)

首先Positional Relation (PR)也就是物体的bounding box是一定有用的，另外又进行了以上三项统计，得到Size Relation (SIR）、Shape Relation (SHR)、Distance Relation (DR)。

### 网络架构

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu10568a7fj30t40fq102.jpg)

### 目标方程

- [12]:

  ![](https://ws1.sinaimg.cn/large/005IsqTWly1fu1066qrnyj30dw04c0tg.jpg)

- 本文加入Region部分的spatial distribution得到region model(S)

  ![](https://ws1.sinaimg.cn/large/005IsqTWly1fu106j9241j30er01zdfw.jpg)

- 最终目标方程：

  ![](https://ws1.sinaimg.cn/large/005IsqTWly1fu106vqqogj30ei07xdgt.jpg)



### 实验结果 

- 由于整个目标方程难以优化，因此每个batch都是进行单独优化的。

- 度量指标仍然是recall@k

  ![](https://ws1.sinaimg.cn/large/005IsqTWly1fu10767rc8j30er07gab4.jpg)

  ![](https://ws1.sinaimg.cn/large/005IsqTWly1fu107ml705j30u807fmz9.jpg)

其实实验结果来看，并没有很大提升作用。
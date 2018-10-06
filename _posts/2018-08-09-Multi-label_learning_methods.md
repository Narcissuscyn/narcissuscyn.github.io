---
layout: post
title:  "Multi label learning methods"
date:   2018-08-09 13:50:39
categories:机器学习
tags:多标签学习
mathjax: true
---
* content
{:toc}
# Multi-label classification

## 1.Formal definitions

1. Learning framework  

   multi-label indicators:

   - label cardinality 

   $$
   LDcard(D)=\frac{1}{m}\sum_{i=1}^{m}|Y_i|
   $$

   - label density 
     $$
     LDen(D)=\frac{1}{|D|}LDcard(D)
     $$

   - label diversity 
     $$
     LDiv(D)=|\{Y|\exists x:(x,Y)\in D\}|
     $$

   - normalized label diversity
     $$
     P LDiv(D) = \frac{1}{|D|} · LDiv(D)
     $$








   Real value function f:
$$
   f：X\times Y\rightarrow \mathbb{R}
$$

   > 	 where f(x, y) can be regarded as the confidence of y ∈ Y being the proper label of x. Specifically, given a multi-label example (x, Y ), f(·, ·) should yield larger output on the relevant label $y ′ ∈ Y$ and smaller output on the irrelevant label $y^{''}\notin   Y $

   multi-label classifier h(·)：
$$
   h(x) = \{y | f(x, y) > t(x), y ∈ Y\} 
$$

   > where t : X → R acts as a `thresholding function` which dichotomizes the label space into relevant and irrelevant label sets 

   

2. key challenge:label correlations 

   - First-order strategy 
   - Second-order strategy
   - High-order strategy 

3. threshold calibration

   > in order to decide the proper label set for unseen instance x (i.e. h(x)), the real-valued output f(x, y) on each label should be calibrated against the thresholding function output t(x) 

   - constant function or inducing t(·) from the training examples 
   - a linear model for t(·) 

## 2.Evaluation Metrics

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu3c8mtezej30iz0amt9z.jpg)

## 3.learning algotithms

![1533789197423](C:\Users\nian.000\AppData\Local\Temp\1533789197423.png)


---
layout: post
title:  "Neural Motifs Scene Graph Pasing With Global Context!"
date:   2018-08-12 13:50:39
categories: 图像理解
tags:RelationshipDetection
mathjax: true
---

* content
{:toc}
### Stacked Motif Networks



**数据分析**

> VG数据库中主要由这3种关系组成: geometric(几何): 50.9%, possessive(所有格): 40.9% semantic(语义): 8.7%，剩下部分为其他.  然后本文首先从数据库中发现衣服、身体部件大多是所有格关系；家具、建筑大多是几何关系；人大多是语义关系的主语，**这些都说明我们在生成SG时有很多`先验`可以利用** 

- [ ]   从这里来看的话，他这个关系分许还是相对粗糙一些，那我那边可以分析得更细致一些，也可以像他这样做一个统计，再得出一些统计结论。

------

`Motifs`

 > motif我翻译成了模板，即关系对(主语类别，关系，宾语类别)中至少2个一样，不区分instance。结果发现，有50%的图片中有长度是2的motif，也就是说某一motif在这些图片中出现一次的情况下，大概率还会出现第二次，就像上图中的elephant has一样，一下出现了4次。这启发我们**在预测关系的时候要考虑全局上下文信息，即要考虑全局中出现的motif，它们之间也是有联系的** 

- [ ]  我觉得它这种Motifs不是很好，他这个Motifs是指图像中出现的重复的一些`子结构`，其实这种结构如果从生成场景图来看，也就是生成的时候直接生成所有的关系，或许有些用，如果单从一个独立的关系来看，是不具有很大作用的,而且相对于ECCV2018的文章[1]来看，他这个Motifs作为全局信息设计的并不是很好，不过，这也算是利用了`全局的context`。从他这些数据分析来看，我更加坚定了我之前的一些想法：把关系分成多类，每一类都应该有他们共享的特征，应该单独为每一类设计特征提取器！

------

------

**网络设计**

> the key challenge in modeling scene graphs lies in devising an efficient mechanism to encode the global context that can directly inform the local predictors(i.e., objects and relations)   
>
> strong independence assumptions in local predictors limit the quality of global predictions.Instead, our model predicts graph elements by staging **bounding box predictions**, **object classifications**, and **relationships** such that the **global context encoding of all previous stages establishes rich context for predicting subsequent stages**    

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu7wyl2op6j30vm0inaoj.jpg)

1) 经过Faster-RCNN检测的结果，由一个bi-LSTM提取物体的context:
$$
C = biLSTM([f_i;W_1l_i]_{i=1,...,n})
$$
2) 再用一个LSTM提取label的context:`use an LSTM to decode a category label for each contextualized representation in C `
$$
hi = LSTM_i ([c_i; \hat{o_i}−1])
$$
经过decoding预测物体label:
$$
\hat{o_i} = argmax (W_o h_i) ∈ R^{|C|} (one-hot)
$$
3) 最后一个Bi-LSTM用于提取predicate的context:


$$
D = biLSTM([c_i;W_2\hat{o_i}]_{i=1,...,n}),
$$
经过decoding预测predicate。

这里面C,$\hat{o}$,D分别对用了bounding box,label,predicate三个阶段的global context

------

------

**实验tricks**

- [ ]  Faster-RCNN进行物体检测，在VG上（150 objects，50 predicates）上物体检测的mAP为20,而我的只达到了14，同样在使用VGG的情况下。可以参考下他们的设置：

  > We optimize the detector using SGD with momentum on 3 Titan Xs, with a batch size of b = 18, and a learning rate of lr = 1.8 · 10−2 that is divided by 10 after validation mAP plateaus. For each batch we sample 256 RoIs per image, of which 75% are background. The detector gets 20.0 mAP (at 50% IoU) on Visual Genome 

- [ ]   解决梯度消失问题：为所有LSTM采用highway connections；

- [ ]  损失函数依旧采用的cross entropy loss,而且`In cases with multiple edge labels per directed edge (5% of edges), we sample the predicates. `不太懂这里的到底用意是什么，只提了这么一句；

  实际上并不是多标签分类：`We follow previous work in enforcing that for a given head and tail bounding box, the system must not output multiple edge labels `

- [ ]  用NMS阈值设为0.3来fine-tune训练好的模型，以解决在测试时由于检测结果不精准带来的不好测试结果；

- [ ]  low-quality RoI pairs 会导致训练不稳定，因此在pair的选取上做了一些阈值的限定；

- [ ]  测试中效果不好的例子有：

  - 谓词标签模糊的（“wearing","wears"）
  - 检测错误的（一个物体检测错误，会导致所有与之相关的关系都预测错误）



**实验设置**

1. 统计模型：用于说明物体检测的重要性。
   - FREQ模型：使用预训练的detector用来预测object label；用统计概率得到predicate probabilities.
   - FREQ+OVERLAP：在统计谓词频率的时候，要求两个物体有overlap。
2. Stanford模型：在[47]上面修改的，消息传递结构用的是本文的motif。
3. Ablations:
   - MotifNet-NoContext:直接用object的label embedding($d_i,d_j$)输入公式6,就是不要$f_{i,j}$.然后利用公式7预测关系。说明vision feature的重要。
   - MotifNet-different roi ordering scheme：完整的网络



**实验结果**

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu819pw1w6j30tk0cw78y.jpg)



**相关工作**

1. context

   - Our approach is most closely related to work that` models object co-occurrence using graphical models` to combine many sources of contextual information [33, 11, 26, 10] 

   - `Actions` and relations have been a particularly fruitful source of context [30, 50] 

   - `human-object interactions` [48, 3] 

   - `object layouts` can provide sufficient context for captioning COCO images [52, 28] 

2. structured model

   - `Deep sequential models` have demonstrated strong performance for tasks such as captioning [4, 9, 45, 18] and visual question answering [1, 37, 53, 12, 8],`multilabel classification` [46] 
   - `graph linearization` for object detections [52], language parsing [44], generating text from abstract meaning graphs [21] 
   - `our`:RNNs provide effective context for consecutive detection predictions. 

3. Scene Graph Methods 

   - `explore the role of priors` ,we allow our model to directly learn to use scene graph priors effectively 
   - 一些图模型的问题： recent graph-propagation methods were applied but converge quickly and bottle neck through edges, significantly limiting information exchange [47, 25, 6, 23]. 
   - VTransE等新卷积特征、新目标方程网络。

**参考文献**

[1] Yang J, Lu J, Lee S, et al. Graph R-CNN for Scene Graph Generation[J]. arXiv preprint arXiv:1808.00191, 2018. 



**补充材料**

最后的材料里面进行了多个论文模型的对比，其中有一处值得注意的是解除了场景图的限制，使一对物体可以有多条边:(这个地方并不是很理解，因为实现的方案中并没有采取多标签的分类，按我的理解的话，可能是一对物体可以取多个结果，而训练的时候任然是on-hot的？？？)

> Omitting graph constraints, namely, allowing a headtail pair to have multiple edge labels in system output.` We hypothesize that omitting graph constraints should always lead to higher numbers（performance）, since the model is then allowed multiple guesses for challenging objects and relations`    
>
> As expected, removing graph constraints significantly increases reported performance and both predicate detection and phrase detection are significantly less challenging than predicate classification and scene graph detection, respectively.    
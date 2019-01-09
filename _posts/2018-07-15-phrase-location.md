---
layout: post
title:  "论文阅读 Phrase Localization and visual relationship detection with comprehensive Image-language cues"
date:   2018-07-15 17:40:41
categories: 图像理解
tags: SceneGraphGeneration
mathjax: true
author: Narcissus
---

* content
{:toc}
这篇文章主要是对phrase location（就是说输入图像和image caption来定位objects）来做的，后面又应用这个来做了一个关系检测的实验，我觉得这很可能是在后期做的一个补充。检测结果相对于前面看的几篇文章都要好。他这个创新点，主要是用了多种cues，并自己设计了有效的损失函数等，phrase lacation这里没有细看。

用了字幕数据，通过nlp进行parse sentence，phrase的解析得到entities、relationships、以及一些其他的linguistic cues。

 

# 1.文章思想

输入image和caption，用nlp tools处理代指关系，提取caption中的noun phrase和pair of entities。并做以下处理：

- 1）对于noun phrase（如red umbrella），根据appearance、size、position、attributes（形容词）等来为每一个候选的box计算一个single-phrase      cue scores；
- 2）对于pair of entities（man carries a      baby），则通过spatial model来对相应的candidate boxes计算score；
- 3）对于actio类的phrase，建立一个subject-verb和verb-object的 appearance models；
- 4）对于people、clothing、body parts之间的关系，phrase通常用来描述individuals，而且也是现有方法比较难定位其中的物体，因此对于这种phrase常常给与spetial treatment。

 

![Input Sentence and Image  A man carries  a baby  under a red  and blue umbrella  next to a woman  Cues  I) Entities  2 CandZate Positi  3) Candidate BOX Size  4)  S) Adjectives  6 Subject - Verb  7) Verb — Object  8 Verbs  Examples  man,  woma n.  man —4 person  person  man.  man,  . under.  man. to.  ) Cloth ing & Body P*tS  Figure 1: Left: an image and caption, together with ground truth bounding  boxes of entities (C:\Users\nian\OneDrive\NoteBook\_posts\assets\clip_image001.png). Right: a list of all the cues used by our  system, with colTesponding phræses from the sentence. ](https://ws1.sinaimg.cn/large/007tAU6Agy1fz0eyq1hl0j30dq09z785.jpg)

 

# 2.文章用了多种组合的特征：

现有的一些工作要么只定位那些attribute phrase中的物体，要么只解决了possessive pronouns 等代指的关系，而本文：

- 1）we use pronoun cues   to the full extent by performing pronominal coreference ；
- 2）ours is the only work in this area incorporating      the visual aspect of verbs. 
- 3） with a larger set of  cues, learned combination weights, and a global optimization method for simultaneously localizing all the phrases in a sentence （就是说能同时解决一个sentence中的所有phrases）

![](https://ws1.sinaimg.cn/large/007tAU6Agy1fz0f1gpu7yj30ua0bkgrn.jpg)

 

# 3.从sentence 解析 phrases

linguistic cues corresponding to adjectives, verbs, and prepositions must be extracted from the captions using NLP tools. hey will be translated into visually relevant constraints for grounding. And we will learn specialized detectors for adjectives, subject-verb, and verb-object relationships ，we will train classifiers to score

pairs of boxes based on spatial information 。

这个部分就是根据nlp提取一些形容词、动词、entities、relationships等language cues，具体的有点感觉，考虑了多种特殊的情况。

 

# 4.损失函数的构建

single phrase cues (SPC) measure the compatibility of a given phrase with a candidate bounding box, and phrase pair cues (PPC) ensure that pairs of related phrases are localized in a spatially coherent manner 

- single phrase cues

- phrase-pair cues

 

# 5.同时进行多个phrase location的过程（测试过程）

# 6.关系检测任务

这个模型扩展到VRD任务上去，能有较好的zero-shot detection的作用。

用了以下几个CCA 模型，用的是fast rcnn在MSCOCO上面训练的模型：

![ ](https://ws1.sinaimg.cn/large/007tAU6Agy1fz0f087auoj30bw09ewf8.jpg)

# 7.VRD实验结果

![- b 一 h d - ; mitial 12  (C:\Users\nian\OneDrive\NoteBook\_posts\assets\clip_image004.png)' CVA  CCA 4 Sir Position 1837 15 , 一 1 43 7  CCA 4 Sir  一 271  一 - R 50 R 50 ](https://ws1.sinaimg.cn/large/007tAU6Agy1fz0eyq76c4j30bz076dgg.jpg)
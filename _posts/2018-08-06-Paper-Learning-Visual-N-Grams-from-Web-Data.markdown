---
layout: post
title: "Learning Visual N-Granms from web data"
date: 2018-08-06 17:40:41
mathjax: true
---


## 《Learning Visual N-Granms from web data》

* content
{:toc}

## 一、论文理解
#### 文章思想

	利用弱标签的web数据，建立language与image的Visual n-gram model.能够预测与图像内容相关的phrases，这篇文章的主要贡献在于损失函数，其来源为nlp里面的n-gram模型。
	
	关于loss函数：given an image I, assign a likelihood p(w|I) to each possible phrase (n-gram) w。we develop a novel, differentiable `loss function` that optimizes trainable parameters for `frequent n-grams`, whereas for `infrequent n-grams`, the loss is dominated by the predicted likelihood of smaller `“sub-grams”`.

#### 相关工作

- learning from weakly supervised web data

  本文采用了和[28]一样的弱监督训练数据，但是不同于它的是不仅仅只考虑单个词，而是考虑了n-gram。数据来自图像分享网站：image-comment。

- Relating image content and language    

  没有采用RNN，而是采用了bilinear model，它也能根据给定图像输出phrases的概率，并把相关的phrases组合成caption。而与其他类似文章所不同的是：本文能处理大量的visual concepts，而不仅仅限制于flickr类似的数据集上面的评论内容，更加能用于实际问题。此处与[40]最相关，但是使用了一个端到端的弱监督训练方法。

- Language models    

  使用了n-gram的语言模型，并使用了Jelinek Mercer smoothing[26] 。

  `n-gram models count the frequency of n-grams in a text corpus to produce a distribution over phrases or sentences, our model measures phrase likelihoods by evaluating inner products between image features and learned parameter vectors.    `

#### 数据集及预处理

- train: YFCC100M dataset，其标注作为弱标签

  comments:We applied a simple language detector to the dataset to select only images with English user comments, leaving a total of 30 million examples for training and testing. We preprocessed the text by removing punctuations, and we added [BEGIN] and [END] tokens at the beginning and end of each sentence.     

  images:rescaling them to 256×256 pixels (using bicubic interpolation), cropping the central 224× 224, subtracting the mean pixel value of each image, and dividing by the standard deviation of the pixel values    

- 训练词典：英语短语：1-5 grams

  the `smoothed visual n-gram models` are trained and evaluated on all n-grams in the dataset, even if these n-grams are not in the dictionary. However, whereas the probability of `indictionary n-grams` is primarily a function of parameters that are specifically tuned for those n-grams, the probability of `out-of-dictionary` n-grams is composed from the probability of smaller in-dictionary n-grams (details below). 

  `不在词典中的短语的概率是通过短语的子短语计算的，子短语肯定是会存在于词典中的。  `

#### 损失函数函

  - $ \phi(I,\theta) ​$是CNN特征提取网络
  - I是图像
  - denote the n-gram dictionary that our model uses by `D` and `a comment containing K words `by $ w ∈ [1, C]^K $ , where `C is the total number of words in the (English) language`. We denote the n-gram that ends at the i-th word of comment w by $w^i_{i-n+1}$ and the ith word in comment w by $w_i^i$.`omit the sum over all image-comment pairs in the training / test data when writing loss functions(也就是说下面损失函数的写法就只包含了一张图像上的一个评论的损失)`. （也就是说训练数据的n-gram词典记为D，每个图像的评论长度固定为K，一条评论为这个语言的单词词典C中的单词组成）
  - 预测的分布是一个 n-gram embedding matrix $ E ∈ R^{D×|D|}$ （也就是英语短语词典中每个短语的词向量）  
##### **Naive n-gram loss**

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu126trr18j30i60hzq7j.jpg)

由于此模型不是一个条件概率，不能进行语言模型的建模，因此加入一个back-off model[6]，(这里也就是对评论语句里的每个单词建立语言模型：n-gram模型，这里对损失函数的建立相当于提供了弱监督信息-由用户评论得到的语言模型):

$$ p(w_i^i|w_{i-n+1}^i)\propto \begin{equation}   \left\{   \begin{array}{**rcl**} p_{obs}(w_i^i|w_{i-n+1}^i),if w_{i-n+1}^i \in D  & ,\\ \lambda p(w_i^i|w_{i-n+2}^i),otherwise \end{array} \right.   \end{equation}  $$

##### **Jelinek-Mercer loss**

![](https://ws1.sinaimg.cn/large/005IsqTWly1fu12klgeg3j30kg0eudjy.jpg)

此处避免了naive 函数的两个弊端。此处提出的Jelinek-Mercer smoothing方法对于E和$\theta$是可导的。因此loss就可以通过卷积网络回传。

#### 训练

- CNN: residual network [23] with 34 layers    
- follow [28] and perform stochastic gradient descent over outputs [4]: we `only perform the forwardbackward pass for a random subset` (formed by all positive n-grams in the batch) of the columns of E.(原因是全部更新的话，输出量太大)

#### 实验

- phrase-level image tagging

  就是给定图像，输出图像内容相关的phrases（table2）；还对模型进行了复杂度的分析（table1）

  这个地方其实也只做了multi-class classification，而并没有做multi-label classification；

  对于figure1中展示的结果：`We predict n-gram phrases for images by outputting the n-grams with the highest calibrated log-likelihood score for an image`，对词典中的每个gram计算了与该图像是否相关的分数，然后取了分数最高的4个phrases；计算分数的方法就是针对语言模型计算，即根据$p(w_i^i|w_{i-n+1}^{i-1})​$计算。

  下表为在测试集上面的结果：We define `recall@k` as the average percentage of ngrams appearing in the comment that are among the k front ranked n-grams when the n-grams are sorted according to their score under the mode。As a baseline    ，we consider a linear multi-class classifier over n-grams (i.e., using naive n-gram loss)    

  ![](https://ws1.sinaimg.cn/large/005IsqTWly1fu12tmts9ij30j00ak76r.jpg)

- phrased based image retrieval

  实验表明，模型能很好的进行区分visual concepts：

  the model has learned accurate visual representations for n-grams such as “Market Street” and “street market”, as well as for “city park” and “Park City”,our model is able to distinguish visual concepts related to Washington: namely, between the state, the city, the baseball team, and the hockey team    

- relating images and captions

  即给定图像检索相关phrases，注意此文用是数据具有much larger vocabularies than the baseline models    

  `以上两个实验任务自然是无法和专门做检索的工作效果好的。`

- zero-shot transfer

  1)The performance of our models is particularly good on common classes such as those in the `aYahoo dataset` for which many examples are available in the YFCC100M dataset. 

  2)The performance of our models is worse on datasets that involve fine-grained classification such as `Imagenet`, for instance, because YFCC100M contains few examples of specific, uncommon dog breeds    

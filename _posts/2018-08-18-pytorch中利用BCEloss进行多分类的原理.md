---
  layout: post
  title:  "pytorch中利用BCEloss进行多分类的原理"
  date:   2018-08-18 13:50:39
  categories: 机器学习
  tags: 损失函数
  mathjax: true
---

* content
{:toc}
# pytorch中利用BCEloss进行多标签分类的原理

classtorch.nn.BCELoss(weight=None, size_average=True)[[source\]](https://pytorch.org/docs/0.3.0/_modules/torch/nn/modules/loss.html#BCELoss)

Creates a criterion that measures the  Binary Cross Entropy between the target and the output:(i为每一个类别)

![loss(o, t) —  —l/n * log(o[i]) + (1 — t[i]) * log(l o i ](file:///C:/Users/nian.000/AppData/Local/Temp/msohtmlclip1/01/clip_image001.png)

or in the case of the weight argument being  specified:

![loss(o, t) —  —l/n weight[i] * (t[i] * log(o[i]) + (1  — t[i]) * log(l o i ](file:///C:/Users/nian.000/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

This is used for measuring the error of a  reconstruction in for example an auto-encoder. Note that the targets t[i] should  be numbers between 0 and 1.

| weight       | weight ([Tensor](https://pytorch.org/docs/0.3.0/tensors.html#torch.Tensor), optional) – a manual rescaling weight given to the loss of each batch element. If given, has to be aTensor of size “nbatch”. |
| ------------ | :----------------------------------------------------------- |
| size_average | size_average ([bool](https://docs.python.org/2/library/functions.html#bool), optional) – By default, the losses          are averaged over observations for each minibatch. However, if the field size_average is set to False, the losses are instead summed for each minibatch. Default: True |

Shape:

- Input: (N,∗)(N,∗) where * means, any number of       additional dimensions
- Target: (N,∗)(N,∗), same shape as the input

Examples:

```python
>>> m = nn.Sigmoid()
>>> loss = nn.BCELoss()
>>> input = autograd.Variable(torch.randn(3), requires_grad=True)
>>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
>>> output = loss(m(input), target)
>>> output.backward()

```

来自 <<https://pytorch.org/docs/0.3.0/nn.htmlhighlight=bceloss#torch.nn.BCELoss>>



## 1.多标签分类原理

主要是结合sigmoid来使用，经过classifier分类过后的输出为（batch_sizze，num_class）为每个数据的标签,标签不是one-hot的主要体现在sigmoid（output）之后进行bceloss计算的时候：sigmoid输出之后,仍然为（batch_sizze，num_class）大小的，但是是每个类别的分数，对于一个实例$x_i$，它的各个label的分数加起来不一定要等于1，为实例计算一个bceloss，这个bceloss这个实例在每个类上的cross entropy loss加和求平均得到，这里就体现了多标签的思想。最后一个batch的loss是否需要求平均根据参数size_average来确定，如果不求均值的话，就是所有实例的加和，这样进行梯度回传的时候就会加大参数的更新步长，就类似于在learning rate上乘了batch size。



## 2.loss加和平均之后，网络还知道哪个类的loss大，该更新哪个类吗？

答案是肯定的，因为反向传播求梯度的时候，是从loss开始的，loss的计算过程也是需要进行求梯度进行反向传播的。

 
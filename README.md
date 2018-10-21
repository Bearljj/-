# DeepLearning
用来记载一下机器学习的各种心得
# 机器学习
- Numpy
  NumPy是Python语言的一个扩充程序库。支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。Numpy内部解除了Python的PIL(全局解释器锁),运算效率极好,是大量机器学习框架的基础库
  [官方中文文档](https://www.numpy.org.cn/index.html)
  [机器学习三剑客“简书”](https://www.jianshu.com/p/83c8ef18a1e8)
- Pandas
  Pandas是基于Numpy开发出的,专门用于数据分析的开源Python库
  [机器学习三剑客“简书”](https://www.jianshu.com/p/7414364992e4)
- Matplotlib
  Matplotlib 是Python2D绘图领域的基础套件，它让使用者将数据图形化，并提供多样化的输出格式。 
  [机器学习三剑客“简书”](https://www.jianshu.com/p/f2782e741a75)
- PyTorch
  Facebook开发的用于NLP的机器学习庫
  [PyTorch中文教程](http://pytorch.apachecn.org/cn/tutorials/)
- Pyro
  Uber基于PyTorch开发的深度概率编程语言
- FastAi
  [github](https://github.com/fastai/fastai)
- Keras
- Theano
- Caffe
- SciPy是一个开源的Python算法库和数学工具包。 SciPy包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。与其功能相类似的软件还有MATLAB、GNU Octave和Scilab。 SciPy目前在BSD许可证下发布。
- Scikit-learn是一个用于Python编程语言的免费软件机器学习库。 它具有各种分类，回归和聚类算法，包括支持向量机，随机森林，梯度增强，k均值和DBSCAN，旨在与Python数值和科学库NumPy和SciPy互操作。[中文文档](http://sklearn.apachecn.org/cn/0.19.0/)

训练模型建议
> - 训练误差应该稳步减小，刚开始是急剧减小，最终应随着训练收敛达到平稳状态。
- 如果训练尚未收敛，尝试运行更长的时间。
- 如果训练误差减小速度过慢，则提高学习速率也许有助于加快其减小速度。
- 但有时如果学习速率过高，训练误差的减小速度反而会变慢。
- 如果训练误差变化很大，尝试降低学习速率。
- 较低的学习速率和较大的步数/较大的批量大小通常是不错的组合。
- 批量大小过小也会导致不稳定情况。不妨先尝试 100 或 1000 - 等较大的值，然后逐渐减小值的大小，直到出现性能降低的情况。

避免很少使用的离散特征值
最好具有清晰明确的含义
不要将“神奇”的值与实际数据混为一谈
考虑上游不稳定性

>确保第一个模型简单易用
着重确保数据管道的正确性
使用简单且可观察的指标进行训练和评估
拥有并监控您的输入特征
将您的模型配置视为代码：进行审核并记录在案
记下所有实验的结果，尤其是“失败”的结果

[L1/L2正则化的直观理解](https://blog.csdn.net/jinping_shi/article/details/52433975)

[机器学习的优化](https://www.leiphone.com/news/201706/e0PuNeEzaXWsMPZX.html)
[效果超过SGD和Adam，谷歌大脑的「神经网络优化器搜索」自动找到更好的训练优化器](https://www.leiphone.com/news/201709/TgVX79CzBaijULDq.html)

![鞍点处的表现](https://static.leiphone.com/uploads/new/article/740_740/201706/5943a045f2b8d.gif)
![损失平面等高线](https://static.leiphone.com/uploads/new/article/740_740/201706/5943a067842cf.gif)
[吴恩达机器学习材料Github](https://github.com/mbadry1/DeepLearning.ai-Summary)

>Colaboratory 是一个 Google 研究项目，旨在帮助传播机器学习培训和研究成果。它是一个 Jupyter 笔记本环境，不需要进行任何设置就可以使用，并且完全在云端运行。Colaboratory 笔记本存储在 Google 云端硬盘 (https://drive.google.com/) 中，并且可以共享，就如同您使用 Google 文档或表格一样。Colaboratory 可免费使用。[本文介绍如何使用 Google CoLaboratory 训练神经网络。](https://www.jiqizhixin.com/articles/2017-12-28-7)
[工具链接：](https://colab.research.google.com/)
>'''
开源项目：https://github.com/Honlan/DeepInterests，欢迎star，这里可以下载到182页的项目说明文档；项目网盘：https://pan.baidu.com/s/1zQRTR5X9JVUxQKNUZxyibg，这里可以下载到所有项目所涉及的代码、数据和模型等资源，分为完整版和精简版两个版本，前者包括项目所涉及的完整资源，后者只包括最后执行所需的必要文件；介绍文章：https://zhuanlan.zhihu.com/p/34744472，自己写的一篇介绍文章，感兴趣的话可以在这里了解到《深度有趣》的更多介绍。

作者：张宏伦
链接：https://www.zhihu.com/question/35396126/answer/130337514
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''






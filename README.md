# 动态图表示论文汇总
本文总结了动态图表示学习的有关论文，目录如下：

- [Static Graph Representation](#static-graph-representation)
    + [Semi-Supervised Classification with Graph Convolutional Networks](#semi-supervised-classification-with-graph-convolutional-networks)
    + [Inductive representation learning on large graphs](#inductive-representation-learning-on-large-graphs)
- [Other Related Works](#other-related-works)
  * [Heterogeneous Graph/Heterogeneous Information Network](#heterogeneous-graph-heterogeneous-information-network)
  * [Others](#others)
- [Dynamic Graph Representation](#dynamic-graph-representation)
    + [Representation Learning for Dynamic Graphs: A Survey](#representation-learning-for-dynamic-graphs--a-survey)
    + [A Survey on Knowledge Graphs: Representation, Acquisition and Applications](#a-survey-on-knowledge-graphs--representation--acquisition-and-applications)
- [New Works of Dynamic Graph Representation (Updating)](#new-works-of-dynamic-graph-representation--updating-)
    + [Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs](#know-evolve--deep-temporal-reasoning-for-dynamic-knowledge-graphs)
    + [DYREP: LEARNING REPRESENTATIONS OVER DYNAMIC GRAPHS](#dyrep--learning-representations-over-dynamic-graphs)
    + [Context-Aware Temporal Knowledge Graph Embedding](#context-aware-temporal-knowledge-graph-embedding)
    + [Real-Time Streaming Graph Embedding Through Local Actions](#real-time-streaming-graph-embedding-through-local-actions)
    + [dyngraph2vec](#dyngraph2vec)
    + [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](#evolvegcn--evolving-graph-convolutional-networks-for-dynamic-graphs)
    + [Temporal Graph Networks for Deep Learning on Dynamic Graphs](#temporal-graph-networks-for-deep-learning-on-dynamic-graphs)
    + [Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN](#modeling-dynamic-heterogeneous-network-for-link-prediction-using-hierarchical-attention-with-temporal-rnn)
    + [Relationship Prediction in Dynamic Heterogeneous Information Networks](#relationship-prediction-in-dynamic-hetergeneous-information-networks)
    + [Link Prediction on Dynamic Heterogeneous Information Networks](#link-prediction-on-dynamic-heterogeneous-information-networks)
    + [Continuous-Time Relationship Prediction in Dynamic Heterogeneous Information Networks](#continuous-time-relationship-prediction-in-dynamic-heterogeneous-information-networks)
- [Related Datasets](#related-datasets)


## Static Graph Representation
挑选了引用数较高、知名度较大的一些静态图表示学习的工作。

#### Semi-Supervised Classification with Graph Convolutional Networks
* 作者：Thomas N. Kipf, et al. (University of Amsterdam)
* 发表时间：2016
* 发表于：ICLR 2017
* 标签：图神经网络
* 概述：提出了图卷积神经网络的概念，并使用其聚合、激活节点的一阶邻居特征。
* 链接：https://arxiv.org/pdf/1609.02907.pdf
* 相关数据集：
    * Citeseer
    * Cora
    * Pubmed
    * NELL
* 是否有开源代码：有
#### Inductive representation learning on large graphs
* 作者： Hamilton W, et al.(斯坦福大学Leskovec团队)
* 发表时间：2017
* 发表于：Advances in neural information processing systems
* 标签：Inductive Graph Embedding
* 概述：针对以往transductive的方式（不能表示unseen nodes）的方法作了改进，提出了一种inductive的方式改进这个问题，该方法学习聚合函数，而不是某个节点的向量
* 链接：https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
* 相关数据集：
    * Citation
    * Reddit
    * PPI
* 是否有开源代码：有

## Other Related Works
### Heterogeneous Graph/Heterogeneous Information Network

### Others

## Dynamic Graph Representation
该部分包括综述论文，以及一些动态图表示的传统工作。

#### Representation Learning for Dynamic Graphs: A Survey
* 作者：Seyed Mehran Kazemi, et al. (Borealis AI)
* 发表时间：2020.3
* 发表于：JMLR 21 (2020) 1-73
* 标签：动态图表示，综述
* 概述：针对目前动态图表示已有的方法，从encoder/decoder的角度进行了概述，覆盖面很全，是了解动态图研究的必读工作。
* 链接：https://deepai.org/publication/relational-representation-learning-for-dynamic-knowledge-graphs-a-survey

#### A Survey on Knowledge Graphs: Representation, Acquisition and Applications
* 作者： Shaoxiong Ji, et al.
* 发表时间：2020
* 发表于：Expert Systems with Applications, 2020
* 关键词：知识图谱，综述
* 概述：本文从知识的表示学习、知识获取，**时态知识图谱**以及知识感知应用等方面做了阐述，内容全面又不失深度，值得一读。
* 链接：https://arxiv.org/pdf/2002.00388.pdf

## New Works of Dynamic Graph Representation (Updating)
挑选了动态图表示领域最近2-3年的工作（2017-2020）。

#### Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs
* 作者： Rakshit Trivedi, et al. (Georgia Institute of Technology)
* 发表时间：2017
* 发表于：PMLR 2017
* 关键词：动态知识图谱
* 概述：作者提出了一套能够在动态演化知识图谱上学习实体表示随时间动态演化的框架。其中采用了基于强度函数的多变量点过程来建模事实的发生概率。作者在两个real-world数据集上对链接预测、实体预测、时间预测与滑动窗口预测等任务进行了评价，验证了该框架的有效性。该论文可以看作DyRep的前置工作。
* 链接：https://arxiv.org/pdf/1705.05742.pdf
* 相关数据集：
    * GDELT
    * ICEWS
* 是否有开源代码：无

#### DYREP: LEARNING REPRESENTATIONS OVER DYNAMIC GRAPHS
* 作者： Rakshit Trivedi, et al. (Georgia Institute of Technology & DeepMind)
* 发表时间：2019
* 发表于：ICLR 2019
* 关键词：CTDG
* 概述：在本文中，作者提出了一套动态图节点表示学习框架，该框架能很好地建模网络的动态演化特征，并能够对unseen nodes进行表示。有对于动态图结构中节点的交互行为，作者将其分为association与communication两种，前者代表长期稳定的联系，网络拓扑结构发生了变化，后者代表短暂、临时的联系。在节点的信息传播方面，作者将节点的信息传播定义为Localized Embedding Propagation/Self-Propagation/Exogenous Drive，分别代表节点邻居的信息聚合传播，节点自身信息传播以及外因驱动（由时间控制）。作者在dynamic link prediction & time prediction任务上对该方法的有效性进行了验证。
* 链接：https://openreview.net/pdf?id=HyePrhR5KX
* 相关数据集：
    * Social Evolution Dataset
    * Github Dataset
* 是否有开源代码：无（有第三方开源代码）

#### Context-Aware Temporal Knowledge Graph Embedding
* 作者： Yu Liu, et al. (昆士兰大学)
* 发表时间：2019
* 发表于：WISE 2019
* 关键词：时态知识图谱，知识表示
* 概述：作者认为现有的knowledge graph embedding方法忽略了时态一致性；时态一致性能够建模事实与事实所在上下文（上下文是指包含参与该事实的所有实体）的关系。为了验证时态知识图谱中事实的有效性，作者提出了上下文选择的双重策略：1、验证组成该事实的三元组是否可信；2、验证这个事实的时态区间是否与其上下文冲突。作者在实体预测/上下文选择任务上证明了方法的有效性。
* 链接：https://link.springer.com/chapter/10.1007/978-3-030-34223-4_37
* 相关数据集：
    * YAGO11k
    * Wikidata12k
* 是否有开源代码：无

#### Real-Time Streaming Graph Embedding Through Local Actions
* 作者： Xi Liu, et al. (德州农工大学)
* 发表时间：2019
* 发表于：WWW 2019
* 关键词：streaming graph
* 概述：本文认为已有的动态图嵌入式学习方法强烈依赖节点属性，时间复杂度高，新节点加入后需要重新训练等缺点。本文提出了streaming graph的概念，提出了一种动态图表示的在线近似算法。该算法能够为新加入图中的节点快速高效生成节点表示，并能够为新加入节点“影响”到的节点更新节点的表示。
* 链接：https://dl.acm.org/doi/abs/10.1145/3308560.3316585
* 相关数据集：
    * Blog
    * CiteSeer
    * Cora
    * Flickr
    * Wiki
* 是否有开源代码：无

#### dyngraph2vec
* 作者： Palash Goyal, et al. (南加州州立大学)
* 发表时间：2020
* 发表于：Knowledge-Based Systems
* 关键词：DTDG
* 概述：本文提出了一种能够捕捉动态图演化的动力学特征，生成动态图表示的方法，并通过AE/RNN/AERNN三种方法进行了实验。基于此，作者设计了一个图embedding生成库GEM
* 链接：https://www.sciencedirect.com/science/article/pii/S0950705119302916
* 相关数据集：
    * SBM dataset
    * Hep-th Dataset
    * AS Dataset
* 是否有开源代码：有


#### EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs
* 作者： Aldo Pareja, et al.(MIT-IBM Watson AI Lab)
* 发表时间：2019
* 发表于：AAAI 2020
* 标签：图卷积网络，DTDG
* 概述：本文不同于传统的DTDG表示学习工作，没有用RNN编码各个snapshot之间的表示，而是使用RNN去编码GCN的参数，从而学习图的演化规律。
* 链接：https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ParejaA.5679.pdf
* 相关数据集：
    * Stochastic Block Model
    * Bitcoin OTC
    * Bitcoin Alpha
    * UC Irvine messages
    * Autonomous systems
    * Reddit Hyperlink Network
    * Elliptic      
* 是否有开源代码：有

#### Temporal Graph Networks for Deep Learning on Dynamic Graphs
* 作者：Rossi, Emanuele, et al.（Twitter）
* 发表时间：2020.6
* 发表于：arXiv
* 标签：动态图表示，CTDG
* 概述：提出了CTDG动态图的一套通用表示框架，并提出了一种能够并行加速训练效率的算法。
* 链接：https://arxiv.org/pdf/2006.10637.pdf
* 相关数据集：
    * Wikipedia（这个数据集是不是开源的Wikidata？论文中无说明）
    * Reddit
    * Twitter
* 是否有开源代码：无

#### Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN
* 作者：Hansheng Xue, Luwei Yang, et al.（澳大利亚国立大学, 阿里巴巴）
* 发表时间：2020.4
* 发表于：arXiv
* 标签：动态图表示，异构图，注意力机制，DTDG
* 概述：本文同时考虑到图的异构性和动态性的特点，对于图的每个时间切片，利用node-level attention和edge-level attention以上两个层次的注意力机制实现异质信息的有效处理，并且通过循环神经网络结合self-attention研究节点embedding的演化特性，并且通过链接预测任务进行试验，验证模型的有效性。
* 链接：https://arxiv.org/pdf/2004.01024.pdf
* 相关数据集：
    * Twitter
    * Math-Overflow
    * Ecomm
    * Alibaba.com
* 是否有开源代码：有(https://github.com/skx300/DyHATR)

#### DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks
* 作者： Aravind Sankar, et al.(UIUC)
* 发表时间：2020
* 发表于：WSDE 2020
* 标签：DTDG，注意力机制
* 概述：作者提出了DYNAMIC SELF-ATTENTION NETWORK机制，通过结构化注意力模块与时态注意力模块对动态变化的节点进行表示。
* 链接：http://yhwu.me/publications/dysat_wsdm20.pdf
* 相关数据集：
    * Enron Email
    * UCI Email
    * MovieLens-10M
    * Yelp      
* 是否有开源代码：有（https://github.com/aravindsankar28/DySAT）

#### Evolving network representation learning based on random walks
* 作者： Farzaneh Heidari, et al.(York University)
* 发表时间：2020
* 发表于：Journal Applied Network Science 2020 (5)
* 标签：DTDG，随机游走
* 概述：针对DTDG动态图的4种演化行为（增加/删除节点，增加/删除边），作者提出了一种在动态图上更新已采样随机游走路径的算法，并设计了网络结构演化程度的Peak Detection算法，从而以较小代价更新不断演化的节点表示。
* 链接：https://appliednetsci.springeropen.com/articles/10.1007/s41109-020-00257-3
* 相关数据集：
    * Protein-Protein Interactions
    * BlogCatalog (Reza and Huan)
    * Facebook Ego Network(Leskovec and Krevl 2014)
    * Arxiv HEP-TH (Leskovec and Krevl 2014)  
    * Synthetic Networks (Watts-Strogatz (Newman 2003) random networks)    
* 是否有开源代码：有（https://github.com/farzana0/EvoNRL）

#### Relationship Prediction in Dynamic Heterogeneous Information Networks
* 作者： Amin Milani Fard, et al.(New York Institute of Technology)
* 发表时间：2019
* 发表于：Advances in Information Retrieval 2019 (4)
* 标签：DTDG，异质信息
* 概述：本文在考虑图动态性的同时，考虑图的异质性，认为不同类型节点对之间的关系自然有所区别，因此提出了动态异质图表示学习，并且做了规范定义。并且提出MetaDynaMix 方法，通过meta-path标注每个节点和边的特征，在此基础上通过矩阵分解得到特征向量，并用于计算关系预测时的概率。
* 链接：https://www.researchgate.net/publication/332257507_Relationship_Prediction_in_Dynamic_Heterogeneous_Information_Networks
* 相关数据集：
    * Publication Network (DBLP+ ACM)
    * Movies Network (IMDB)
* 是否有开源代码：无

#### Link Prediction on Dynamic Heterogeneous Information Networks
* 作者： Chao Kong, et al.(Anhui Polytechnic University)
* 发表时间：2019
* 发表于：Lecture Notes in Computer Science 2019
* 标签：DTDG，异质信息，广度学习，图神经网络
* 概述：本文考虑到动态图相关研究中异质信息缺乏有效的利用，且对于大规模图的表示学习过程中，深度学习方法效率较低，因此提出了一种宽度学习(?)的框架，并且与图神经网络相结合，实现高效的动态异质图表示学习。
* 链接：https://link.springer.com/chapter/10.1007%2F978-3-030-34980-6_36
* 相关数据集：
    * Reddit
    * Stack Overflow
    * Ask Ubuntu
* 是否有开源代码：无

#### Continuous-Time Relationship Prediction in Dynamic Heterogeneous Information Networks
* 作者： SINA SAJADMANESH, et al.(Sharif University of Technology)
* 发表时间：2018
* 发表于：ACM Transactions on Knowledge Discovery from Data (5)
* 标签：CTDG，异质信息
* 概述：本文同时关注到图的动态性与异质性，针对于连续时间的关系预测问题进行了定义，并且提出了一种新的特征抽取框架，通过Meta-Path以及循环神经网络实现对于异质信息与时间信息的有效利用，并且提出NP-GLM框架，用于实现关系预测(预测关系创建的时间节点)。 
* 链接：https://www.researchgate.net/publication/320195531_Continuous-Time_Relationship_Prediction_in_Dynamic_Heterogeneous_Information_Networks
* 相关数据集：
    * DBLP
    * Delicious
    * MovieLens
* 是否有开源代码：无

## Related Datasets

* Social Evolution Dataset
* Github Dataset
* GDELT (Global data on events, location, and tone)
* ICEWS (Integrated Crisis Early Warning System)
* FB-FORUM
* UCI Message data
* YELP
* MovieLens-10M
* SNAP数据集合网站：http://snap.stanford.edu/data/index.html
* SNAP时态数据集合：http://snap.stanford.edu/data/index.html#temporal
* KONECT数据集合网站（部分数据集的edge带有时间戳，可看作时序数据）
* Network Repository：http://networkrepository.com/

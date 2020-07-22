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

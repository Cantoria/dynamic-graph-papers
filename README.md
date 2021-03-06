# 动态图表示论文汇总

本项目总结了动态图表示学习的有关论文，该项目在持续更新中，欢迎大家watch/star/fork！
如果大家有值得推荐的工作，可以在issue中提出要推荐的工作、论文下载链接及其工作亮点。项目中表述有误的部分，也可以在issue中提出。感谢！

引流：【这也是我们的工作，欢迎watch/star/fork】

社交知识图谱专题：https://github.com/jxh4945777/Social-Knowledge-Graph-Papers

目录如下：

- [Static Graph Representation & Analyzing Works](#static-graph-representation---analyzing-works)
    + [node2vec: Scalable Feature Learning for Networks](#node2vec--scalable-feature-learning-for-networks)
    + [Semi-Supervised Classification with Graph Convolutional Networks](#semi-supervised-classification-with-graph-convolutional-networks)
    + [Inductive representation learning on large graphs](#inductive-representation-learning-on-large-graphs)
- [Other Related Works](#other-related-works)
  * [Heterogeneous Graph/Heterogeneous Information Network](#heterogeneous-graph-heterogeneous-information-network)
    + [Heterogeneous Network Representation Learning: Survey, Benchmark, Evaluation, and Beyond](#heterogeneous-network-representation-learning--survey--benchmark--evaluation--and-beyond)
    + [异质信息网络分析与应用综述](#-------------)
    + [Modeling Relational Data with Graph Convolutional Networks](#modeling-relational-data-with-graph-convolutional-networks)
  * [Dynamic & Heterogeneous Graph Representation](#dynamic---heterogeneous-graph-representation)
    + [DHNE: Network Representation Learning Method for Dynamic Heterogeneous Networks](#dhne--network-representation-learning-method-for-dynamic-heterogeneous-networks)
    + [Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN](#modeling-dynamic-heterogeneous-network-for-link-prediction-using-hierarchical-attention-with-temporal-rnn)
    + [Dynamic Heterogeneous Information NetworkEmbedding with Meta-path based Proximity](#dynamic-heterogeneous-information-networkembedding-with-meta-path-based-proximity)
    + [Relationship Prediction in Dynamic Heterogeneous Information Networks](#relationship-prediction-in-dynamic-heterogeneous-information-networks)
    + [Link Prediction on Dynamic Heterogeneous Information Networks](#link-prediction-on-dynamic-heterogeneous-information-networks)
    + [Heterogeneous Graph Transformer](#heterogeneous-graph-transformer)
  * [Others](#others)
    + [A Survey on Knowledge Graphs: Representation, Acquisition and Applications](#a-survey-on-knowledge-graphs--representation--acquisition-and-applications)
- [Dynamic Graph Representation](#dynamic-graph-representation)
    + [Representation Learning for Dynamic Graphs: A Survey](#representation-learning-for-dynamic-graphs--a-survey)
    + [Foundations and modelling of dynamic networks using Dynamic Graph Neural Networks: A survey](#foundations-and-modelling-of-dynamic-networks-using-dynamic-graph-neural-networks--a-survey)
    + [A Survey on Knowledge Graphs: Representation, Acquisition and Applications](#a-survey-on-knowledge-graphs--representation--acquisition-and-applications-1)
    + [Temporal Link Prediction: A Survey](#temporal-link-prediction--a-survey)
    + [Temporal Networks](#temporal-networks)
    + [Motifs in Temporal Networks](#motifs-in-temporal-networks)
    + [动态网络模式挖掘方法及其应用](#--------------)
- [New Works of Dynamic Graph Representation (Updating)](#new-works-of-dynamic-graph-representation--updating-)
    + [Deep Coevolutionary Network: Embedding User and Item Features for Recommendation](#deep-coevolutionary-network--embedding-user-and-item-features-for-recommendation)
    + [Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs](#know-evolve--deep-temporal-reasoning-for-dynamic-knowledge-graphs)
    + [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](#spatio-temporal-graph-convolutional-networks--a-deep-learning-framework-for-traffic-forecasting)
    + [Embedding Temporal Network via Neighborhood Formation](#embedding-temporal-network-via-neighborhood-formation)
    + [Continuous-Time Dynamic Network Embeddings](#continuous-time-dynamic-network-embeddings)
    + [Dynamic Network Embedding by Modeling Triadic Closure Process](#dynamic-network-embedding-by-modeling-triadic-closure-process)
    + [DYREP: LEARNING REPRESENTATIONS OVER DYNAMIC GRAPHS](#dyrep--learning-representations-over-dynamic-graphs)
    + [Learning to Represent the Evolution of Dynamic Graphs with Recurrent Models](#learning-to-represent-the-evolution-of-dynamic-graphs-with-recurrent-models)
    + [Context-Aware Temporal Knowledge Graph Embedding](#context-aware-temporal-knowledge-graph-embedding)
    + [Real-Time Streaming Graph Embedding Through Local Actions](#real-time-streaming-graph-embedding-through-local-actions)
    + [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](#predicting-dynamic-embedding-trajectory-in-temporal-interaction-networks)
    + [dyngraph2vec-Capturing Network Dynamics using Dynamic Graph Representation Learning](#dyngraph2vec-capturing-network-dynamics-using-dynamic-graph-representation-learning)
    + [Temporal Network Embedding with Micro- and Macro-dynamics](#temporal-network-embedding-with-micro--and-macro-dynamics)
    + [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](#evolvegcn--evolving-graph-convolutional-networks-for-dynamic-graphs)
    + [Temporal Graph Networks for Deep Learning on Dynamic Graphs](#temporal-graph-networks-for-deep-learning-on-dynamic-graphs)
    + [Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN](#modeling-dynamic-heterogeneous-network-for-link-prediction-using-hierarchical-attention-with-temporal-rnn-1)
    + [DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks](#dysat--deep-neural-representation-learning-on-dynamic-graphs-via-self-attention-networks)
    + [Evolving network representation learning based on random walks](#evolving-network-representation-learning-based-on-random-walks)
    + [TemporalGAT: Attention-Based Dynamic Graph Representation Learning](#temporalgat--attention-based-dynamic-graph-representation-learning)
    + [Continuous-Time Relationship Prediction in Dynamic Heterogeneous Information Networks](#continuous-time-relationship-prediction-in-dynamic-heterogeneous-information-networks)
    + [Continuous-Time Dynamic Graph Learning via Neural Interaction Processes](#continuous-time-dynamic-graph-learning-via-neural-interaction-processes)
    + [A Data-Driven Graph Generative Model for Temporal Interaction Networks](#a-data-driven-graph-generative-model-for-temporal-interaction-networks)
    + [Embedding Dynamic Attributed Networks by Modeling the Evolution Processes](#embedding-dynamic-attributed-networks-by-modeling-the-evolution-processes)
    + [Learning to Encode Evolutionary Knowledge for Automatic Commenting Long Novels](#learning-to-encode-evolutionary-knowledge-for-automatic-commenting-long-novels)
    + [Generic Representation Learning for Dynamic Social Interaction](#generic-representation-learning-for-dynamic-social-interaction)
    + [Motif-Preserving Temporal Network Embedding](#motif-preserving-temporal-network-embedding)
    + [Motif-Preserving Temporal Network Embedding](#motif-preserving-temporal-network-embedding-1)
    + [Local Motif Clustering on Time-Evolving Graphs](#local-motif-clustering-on-time-evolving-graphs)
    + [INDUCTIVE REPRESENTATION LEARNING ON TEMPORAL GRAPHS](#inductive-representation-learning-on-temporal-graphs)
    + [INDUCTIVE REPRESENTATION LEARNING IN TEMPORAL NETWORKS VIA CAUSAL ANONYMOUS WALKS](#inductive-representation-learning-in-temporal-networks-via-causal-anonymous-walks)
- [Related Datasets](#related-datasets)
- [其他参考资料](#------)


## Static Graph Representation & Analyzing Works
针对静态图表示学习以及静态图分析、挖掘领域，挑选了个人认为值得借鉴的引用数较高、知名度较大的或最近的一些工作。

#### node2vec: Scalable Feature Learning for Networks
* 作者：Grover A, Leskovec J. (University of Amsterdam)
* 发表时间：2016
* 发表于：KDD 2016
* 标签：图表示学习
* 概述：依据表示学习，提出了一套在网络中学习节点连续型表示的方法，取代了传统使用人工定义节点结构化特征的方式（如中心度等）。其指导思想是是最大化节点邻居共现的似然。其另一贡献是在随机游走采样的基础上提出了BFS与DFS结合的灵活采样方法，能够采样到不同的邻居。
* 链接：https://arxiv.org/pdf/1607.00653.pdf
* 相关数据集：
    * BlogCatalog
    * PPI
    * Wikipedia
* 是否有开源代码：有

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
#### Heterogeneous Network Representation Learning: Survey, Benchmark, Evaluation, and Beyond
* 作者： Carl Yang, et al.(UIUC韩家炜团队)
* 发表时间：2020
* 发表于：Arxiv
* 标签：Heterogeneous Network Reprensentation Learning
* 概述：本文是异质图相关研究的综述文章，系统性地梳理了异质图的经典工作以及前沿工作，将已有工作规范到统一的框架内，且提出了异质图表示学习的Benchmark，并且对于经典的异质图方法进行了复现与评测。
* 链接：https://arxiv.org/abs/2004.00216
* 相关数据集：
    * DBLP
    * Yelp
    * Freebase
    * PubMed
* 是否有开源代码：有

#### 异质信息网络分析与应用综述
* 作者： Chuan Shi, et al.
* 发表时间：2020
* 发表于：软件学报
* 标签：Heterogeneous Information Network
* 概述：本文是一篇关于异质信息网络的最新中文综述，对于异质信息网络给出了明确的定义，并且对于现有异质信息网络的从网络结构的角度进行了归类，对于异质信息网络表示学习相关的工作也进行了归类为基于图分解的方法、基于随机游走的方法、基于编码器-解码器的方法以及基于图神经网络的方法。同时本文对于异质信息网络的应用进行了叙述，最后对于异质信息网络的发展提出了展望。
* 链接：http://www.shichuan.org/doc/94.pdf
* 是否有开源代码：有 https://github.com/BUPT-GAMMA/OpenHINE

#### Modeling Relational Data with Graph Convolutional Networks
* 作者： Michael Schlichtkrull, Thomas N. Kipf, et al.
* 发表时间：2018
* 发表于：ESWC 2018
* 标签：Knowledge Graph, Multi Relation, Graph Neural Network
* 概述：本文关注于真实世界图中边的异质性，例如FB15K-237和WN18包含多种类型的边。现有图神经网络GCN无法建模边的异质性，因此本文提出了R-GCN模型，在信息传递时对于不同类型的边使用不同的权值矩阵，同时考虑到在边比较多的情况下矩阵的数目也较多，因此采取了共享权值的方式，将每种类型边的权值矩阵视作多个基的带权加和，以此缩小参数量。对于实验部分，本文在FB15K，和WN18两个数据集上，从实体分类以及连接预测(知识图谱补全)两个实验角度验证了模型的有效性。
* 链接：https://arxiv.org/abs/1703.06103
* 相关数据集：
    * WN18
    * FB15K-237
* 是否有开源代码：有(https://github.com/tkipf/relational-gcn)


### Dynamic & Heterogeneous Graph Representation

本部分为动态异质图表示学习领域的相关文章，其研究对象为动态异质图。

#### DHNE: Network Representation Learning Method for Dynamic Heterogeneous Networks
* 作者： Ying Yin, et al.
* 发表时间：2019
* 发表于：IEEE Access
* 标签：CTDG，异质信息，动态信息， random walk
* 概述：本文同时考虑到图的异质性与动态性，通过构建Historical-Current图将中心节点的历史邻居信息与当前邻居信息进行拼接，并在此基础上进行Random Walk采样，通过Skip-Gram更新节点在当前时间的向量表示。本文在包含时间信息的DBLP和Aminer数据集上通过节点分类的下游任务验证了模型的有效性。
* 链接：https://ieeexplore.ieee.org/document/8843962
* 相关数据集：
    * AMiner
    * DBLP
* 是否有开源代码：有

#### Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN
* 作者： Hansheng Xue, et al.
* 发表时间：2020
* 发表于：ArXiv
* 标签：CTDG，异质信息，动态信息， 图神经网络
* 概述：本文提出一个能够同时学习图中动态信息和异质信息的框架DyHATR，通过类似于HAN的异质图神经网络建模每个时间步上节点的表示，其中通过分层注意力机制同时关注到聚合信息时不同节点的重要性，以及不同Meta-path的重要性。在对于每个时间切片图中学到节点的表示基础上，通过RNN来建模节点表示的演化。本文通过Link Prediction实验验证了模型的有效性。
* 链接：https://ieeexplore.ieee.org/document/8843962
* 相关数据集：
    * Twitter
    * Math-Overflow
    * Ecomm
* 是否有开源代码：有(https://github.com/skx300/DyHATR)

#### Dynamic Heterogeneous Information NetworkEmbedding with Meta-path based Proximity
* 作者： Xiao Wang, et al.
* 发表时间：2020
* 发表于：TKDE
* 标签：DTDG，异质信息，动态信息， 矩阵分解
* 概述：对于动态异质图，本文提出一种新的增量式更新方法，用于在考虑图演化的情况下节点向量表示的更新。首先本文对于静态异质图的表示学习，提出了新的StHNE模型，能够同时考虑到一阶邻居相似性以及二阶邻居相似性用于作为节点表示的参照；在此基础上，对于动态演化的异质图，本文提出DyHNE模型，将图的演化转化成特征值和特征向量的变化，并且据此提出了一套新的增量式更新的方法，用于更新节点的表示。本文通过节点分类以及关系预测验证了模型的有效性。
* 链接：https://yuanfulu.github.io/publication/TKDE-DyHNE.pdf
* 相关数据集：
    * Yelp
    * DBLP
    * AMiner
* 是否有开源代码：有(https://github.com/rootlu/DyHNE)

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

#### Heterogeneous Graph Transformer
* 作者： Ziniu Hu, et al. (UCLA Yizhou Sun团队)
* 发表时间：2020
* 发表于：WWW 2020
* 标签：Heterogeneous Network Reprensentation Learning, Transformer, Multi-Head Attention
* 概述：考虑到已有异质图的研究存在以下几点局限：1. 需要人工设计Meta-path；2.无法建模动态信息；3.对于大规模的异质图，缺乏有效的采样方式。针对于以上三点，本文首选给出Meta Relation的概念，直接建模相连的异质节点，基于此设计了类Transformer的网络结构用于图表示学习。考虑到异质图的动态特性，本文提出了RTE编码方式，用于建模异质图的动态演化。考虑到大规模异质图上网络的训练，本文提出了HGSampling方式，用于均匀采样不同类型的节点信息，以实现高效的图表示学习。
* 链接：https://arxiv.org/abs/2003.01332
* 相关数据集：
    * OAG
* 是否有开源代码：有 https://github.com/acbull/pyHGT

### Others
#### A Survey on Knowledge Graphs: Representation, Acquisition and Applications
* 作者：Shaoxiong Ji, Shirui Pan, Erik Cambria, Senior Member, IEEE, Pekka Marttinen, Philip S. Yu, Fellow IEEE
* 发表时间：2020
* 发表于：Arxiv
* 标签：Knowledge Graph, Representation Learning
* 概述：本文是一篇知识图谱领域的前沿综述，文中给出了知识图谱的具体定义，并且从知识获取、知识表示、动态知识图谱、知识图谱的应用等多个角度围绕知识图谱技术进行了讨论。同时文章还对于知识图谱未来的发展提出了展望。
* 链接：https://arxiv.org/abs/2002.00388
* 是否有开源代码：无


## Dynamic Graph Representation
该部分包括综述论文，以及一些动态图分析与挖掘、动态图表示的传统工作。



#### Representation Learning for Dynamic Graphs: A Survey
* 作者：Seyed Mehran Kazemi, et al. (Borealis AI)
* 发表时间：2020.3
* 发表于：JMLR 21 (2020) 1-73
* 标签：动态图表示，综述
* 概述：针对目前动态图表示已有的方法，从encoder/decoder的角度进行了概述，覆盖面很全，是了解动态图研究的必读工作。
* 链接：https://deepai.org/publication/relational-representation-learning-for-dynamic-knowledge-graphs-a-survey

#### Foundations and modelling of dynamic networks using Dynamic Graph Neural Networks: A survey
* 作者：Joakim Skarding, et al. (University of Technology Sydney)
* 发表时间：2020.5
* 发表于：arXiv
* 标签：动态图表示，综述，动态图神经网络
* 概述：该文侧重于从图神经网络的角度与具体任务的角度去讲述目前动态网络的研究方向。在第二章中，作者将动态图的有关定义整理为体系，从3个维度（时态粒度、节点动态性、边持续的时间）上，分别定义了8种动态网络的定义。在第三章中，阐述了编码动态网络拓扑结构的深度学习模型；在第四章中，阐述了被编码的动态网络信息如何用于预测，即动态网络的解码器、损失函数、评价指标等。在最后一章，作者阐述了动态图表示、建模的一些挑战，并对未来的发展方向进行了展望。
* 链接：https://arxiv.org/abs/2005.07496

#### A Survey on Knowledge Graphs: Representation, Acquisition and Applications
* 作者： Shaoxiong Ji, et al.
* 发表时间：2020
* 发表于：Expert Systems with Applications, 2020
* 关键词：知识图谱，综述
* 概述：本文从知识的表示学习、知识获取，**时态知识图谱**以及知识感知应用等方面做了阐述，内容全面又不失深度，值得一读。
* 链接：https://arxiv.org/pdf/2002.00388.pdf

#### Temporal Link Prediction: A Survey
* 作者： Divakaran A, et al.
* 发表时间：2019
* 发表于：New Generation Computing (2019)
* 关键词：时态链接预测，综述
* 概述：从离散动态图（DTDG）的角度出发，本文针对时态链接预测任务给出了相关定义，并从实现方法的角度出发，构建了时态链接预测的分类体系，分别从矩阵分解/概率模型/谱聚类/时间序列模型/深度学习等不同方法实现的模型进行了比较与论述。文章还列举出了时态链接预测任务的相关数据集（论文互引网络、通讯网络、社交网络、人类交往网络数据等）。最后，文章对时态链接预测任务的难点进行了展望。
* 链接：https://link.springer.com/article/10.1007%2Fs00354-019-00065-z


#### Temporal Networks
* 作者： Holme P, Saramäki J.
* 发表时间：2012
* 发表于：Physics reports, 2012
* 关键词：时态网络，综述
* 概述：这篇论文是一篇时态网络的经典综述论文。论文中给出了时态网络的三种形式，并且从时态网络的拓扑结构的衡量方法、将时态数据表示为静态图、时态网络的一些模型、时态网络上的传播动力学和区间模型等进行了论述。最后，文章对时态网络的未来发展趋势进行了展望。
* 链接：https://link.springer.com/article/10.1007%2Fs00354-019-00065-z


#### Motifs in Temporal Networks
* 作者： Ashwin Paranjape, et al. 
* 发表时间：2017
* 发表于：WSDM, 2017
* 关键词：时态网络，motif
* 概述：该文将传统图分析中的motif概念引入时态网络中，认为时态网络中的motif是网络中的最基本构成单位，定义了Temporal network motifs与时间间隔关联的δ-temporal motifs的概念；并利用时态网络上的motif分析时态网络上的演化交互规律。此外，作者设计了一种快速计算时态网络中不同类型motif数目的算法，能够快速分析某个时态网络的演化特性。
* 链接：https://dl.acm.org/doi/abs/10.1145/3018661.3018731


#### 动态网络模式挖掘方法及其应用
* 作者： 高 琳, 杨建业, 覃桂敏（西安电子科技大学）
* 发表时间：2013
* 发表于：软件学报
* 关键词：动态网络，模式挖掘
* 概述：该论文以动态网络为对象，对动态网络的拓扑特性分析、社团结构挖掘、子图模式挖掘与模式预测相关的模型和方法进行了综述、比较和分析，在应用层面，描述了生物网络以及社会网络的动态模式。
* 链接：http://www.jos.org.cn/ch/reader/create_pdf.aspx?file_no=4439&amp;journal_id=jos

## New Works of Dynamic Graph Representation (Updating)
挑选了动态图表示领域最近2-3年的工作（2017-2020）。

#### Deep Coevolutionary Network: Embedding User and Item Features for Recommendation
* 作者： Hanjun Dai, et al. (Georgia Institute of Technology)
* 发表时间：2017
* 发表于：KDD 2017
* 关键词：动态演化网络，推荐系统
* 概述：该论文首次将时态点过程与深度学习相结合，针对推荐系统中的user-item时态交互网络中两类节点互相演化的特点，依据点过程（Point Process）理论，提出了一套能够依据交互过程，不断迭代更新user/item节点表示的框架。其中，框架主体采用了两套RNN模型损失函数采用了基于Rayleigh process的强度函数（Intensity function）的联合非负似然概率函数，其包括发生概率（happened probability）与生存概率（survival probability）组成。作者在两个real-world数据集上对链接预测、实体预测、时间预测与滑动窗口预测等任务进行了评价，验证了该框架的有效性。
* 链接：https://arxiv.org/pdf/1609.03675.pdf
* 相关数据集：
    * IPTV
    * Yelp
    * Reddit
* 是否有开源代码：无

#### Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs
* 作者： Rakshit Trivedi, et al. (Georgia Institute of Technology)
* 发表时间：2017
* 发表于：PMLR 2017
* 关键词：动态知识图谱,knowledge
* 概述：作者提出了一套能够在动态演化知识图谱上学习实体表示随时间动态演化的框架。其中采用了基于强度函数的多变量点过程来建模事实的发生概率。作者在两个real-world数据集上对链接预测、实体预测、时间预测与滑动窗口预测等任务进行了评价，验证了该框架的有效性。该论文可以看作DyRep的前置工作。
* 链接：https://arxiv.org/pdf/1705.05742.pdf
* 相关数据集：
    * GDELT
    * ICEWS
* 是否有开源代码：无

#### Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
* 作者： Bing Yu, et al. (Peking University)
* 发表时间：2018
* 发表于：IJCAI 2018
* 关键词：时空图，交通流量预测，DTDG
* 概述：作者将交通流量预测建模为时空图（Spatial- Temporal Graph）的形式，设计了一种能够在时空图上进行运算的GCN架构模型STGCN，其参数量较于传统的序列预测模型（RNN等）大大减少，能够以更快的速度运算。作者在两个交通流量数据集上进行了指标性能与时间运算性能的比较，证明了STGCN的优越性。总体来说，这是一篇偏工程应用的论文。
* 链接：https://www.ijcai.org/Proceedings/2018/0505.pdf
* 相关数据集：
    * BJER4
    * PeMSD7
* 是否有开源代码：无


#### Embedding Temporal Network via Neighborhood Formation
* 作者：Yuan Zuo, et al. (Beihang University)
* 发表时间：2018
* 发表于：KDD 2018
* 关键词：CTDG，点过程，Hawkes Process，时态网络（Temporal Network）
* 概述：传统的DTDG方法将时态网络按照固定的时间点切片为快照表示的静态图模式，其忽略了时态网络中的连边是以互动事件流的形式形成的，而非在某一时间点同时形成。节点的邻居是随时间逐步形成的，且不同时间形成的邻居对该节点有着不同的影响。作者提出了基于霍克斯过程的时序网络嵌入算法，该算法利用Hawkes过程建模节点邻居的序列化产生过程，并利用attention机制建模不同时期的历史邻居对节点的影响力。作者在节点分类、链接预测以及embedding可视化等任务上进行了实验。
* 链接：https://dl.acm.org/doi/pdf/10.1145/3219819.3220054
* 相关数据集：
    * DBLP
    * Yelp
    * Tmall
* 是否有开源代码：无

#### Continuous-Time Dynamic Network Embeddings
* 作者： Giang Hoang Nguyen, et al. (Worcester Polytechnic Institute)
* 发表时间：2018
* 发表于：WWW 2018
* 关键词：动态图表示，temporal random walk
* 概述：依据deepwalk与node2vec等模型的启发，作者基于动态图的性质，提出了temporal random walk的概念，即在一条随机游走路径上，从起始节点到终止节点，连边的时态信息依次递增。针对边上存在时态信息的问题，作者提出了unbiased/biased采样算法。采样后的路径将会蕴含动态图中的时态依赖信息。作者在多个动态图数据集上做了实验，并与Deepwalk/Node2vec/LINE等静态图表示学习算法进行了对比。
* 链接：https://dl.acm.org/doi/abs/10.1145/3184558.3191526
* 相关数据集：
    * ia-contact
    * ia-hypertext09
    * ia-enron-employees
    * ia-radoslaw-email
    * ia-email-eu
    * fb-forum
    * soc-bitcoinA
    * soc-wiki-elec
* 是否有开源代码：有一个第三方复现版本https://github.com/Shubhranshu-Shekhar/ctdne

#### Dynamic Network Embedding by Modeling Triadic Closure Process
* 作者： Lekui Zhou, et al. (Zhejiang University)
* 发表时间：2018
* 发表于：AAAI 2018
* 关键词：动态图表示，DTDG
* 概述：作者依据动态网络的特性，提出了依据triad结构建模动态图演化模式的方法DynamicTraid。三元组（Triad）演化的过程就是三个节点中两个互不链接的节点之间建立链接，形成一个闭合三元组的过程。作者在几个不同的真实业务场景（电信欺诈，贷款偿还等）数据集中做了实验，证明了模型的有效性。
* 链接：http://yangy.org/works/dynamictriad/dynamic_triad.pdf
* 相关数据集：
    * Mobile
    * Loan
    * Academic
* 是否有开源代码：有（https://github.com/luckiezhou/DynamicTriad ）

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
* 是否有开源代码：无（有第三方实现的开源代码）

#### Learning to Represent the Evolution of Dynamic Graphs with Recurrent Models
* 作者： Aynaz Taheri, et al. (UIUC)
* 发表时间：2019
* 发表于：ICLR 2019
* 关键词：DTDG
* 概述：本文提出了一种适用于离散型动态网络的门限图神经网络模型DyGGNN，用于动物行为（animal behaviour）的预测任务。具体地，该框架遵循编码-解码器构造，encoder使用GGNN编码每一个离散图，并利用LSTM编码当前时刻及其前T步的历史序列数据，从而得到整个图的编码向量；decoder使用一个LSTM结构来解码图在每一时间步的拓扑结构。
* 链接：https://dl.acm.org/doi/10.1145/3308560.3316581
* 相关数据集：
    * baboon data
* 是否有开源代码：无

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

#### Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks
* 作者： Srijan Kumar, et al. (斯坦福大学，Jure团队)
* 发表时间：2019
* 发表于：KDD 2019
* 关键词：CTDG，user-item dynamic embedding
* 概述：这篇论文解决的问题是建模user-item之间的序列互动问题。而表示学习能够为建模user-item之间的动态演化提供很好的解决方案。目前工作的缺陷是只有在user作出变化时才会更新其表示，并不能生成user/item未来的轨迹embedding。因此，作者设计了JODIE（Joint Dynamic User-Item Embeddings），其包括更新部分与预测部分。更新部分由一个耦合循环神经网络（coupled recurrent neural network）学习user与item未来轨迹。其使用了两个循环神经网络更新user/item在每次interaction的表示，还能表示user/item未来的embedding变化轨迹（trajectory）。预测部分由一个映射算子组成，其能够学习user在未来任意某个时间点的embedding表示。为了让这个方法可扩展性更强，作者提出了一个t-Batch算法，能够创建时间一致性的batch（time-consistent batch），且能够提升9倍训练速度。为了验证方法的有效性，作者在4个实验数据集上做了实验，对比了6种方法，发现在预测未来互动（predicting future interaction）任务上提升了20%，在状态变化预测（state change prediction任务上提升了12%）
* 链接：https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf
* 相关数据集：
    * Reddit
    * Wikipedia
    * Last FM
    * MOOC
* 是否有开源代码：有(https://snap.stanford.edu/jodie/)

#### dyngraph2vec-Capturing Network Dynamics using Dynamic Graph Representation Learning
* 作者： Palash Goyal, et al. (南加州州立大学)
* 发表时间：2020
* 发表于：Knowledge-Based Systems
* 关键词：DTDG
* 概述：本文首先针对动态图表示学习进行了定义，即：学习到一个函数的映射，这个映射能将每个时间点的图中节点映射为向量y，并且这个向量能够捕捉到节点变化的时态模式。基于此，作者提出了一种能够捕捉动态图演化的动力学特征，生成动态图表示的方法，本质上是输入为动态图的前T个时间步的snapshot，输出为T+1时刻的图嵌入式表达。在实验中，作者采用了AE/RNN/AERNN三种编码器进行了实验。此外，作者设计了一个图embedding生成库DynamicGEM。
* 链接：https://www.sciencedirect.com/science/article/pii/S0950705119302916
* 相关数据集：
    * SBM dataset
    * Hep-th Dataset
    * AS Dataset
* 是否有开源代码：有(https://github.com/palash1992/DynamicGEM)

#### Temporal Network Embedding with Micro- and Macro-dynamics
* 作者： Yuanfu Lu, et al. (德州农工大学)
* 发表时间：2019
* 发表于：CIKM 2019
* 关键词：micro/macro dynamic, Temporal Point Process
* 概述：作者提出了从微观/宏观两种层级建模动态网络中节点演化规律，并能够在节点表示中学习到这种规律。微观更偏向于捕捉具体边对形成过程
宏观更偏向于从网络动力学挖掘网络演变的规律，最终生成节点的表示。论文作者设计了多种实验，并依据实验验证了模型在准确性（分别是Network Reconstruction与Node Classification）、动态性（Temporal Node Recommendation与Temporal Link Prediction）、可扩展性（规模预测、趋势预测）等性能上的表现，证明了模型的有效性。
* 链接：https://dl.acm.org/doi/abs/10.1145/3308560.3316585
* 相关数据集：
    * Eucore
    * DBLP
    * Tmall
* 是否有开源代码：有（https://github.com/rootlu/MMDNE ）
 
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
* 是否有开源代码：有（https://github.com/IBM/EvolveGCN ）

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
* 是否有开源代码：有(https://github.com/aravindsankar28/DySAT)

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
* 是否有开源代码：有(https://github.com/farzana0/EvoNRL)

#### TemporalGAT: Attention-Based Dynamic Graph Representation Learning
* 作者： Ahmed Fathy and Kan Li(Beijing Institute of Technology)
* 发表时间： 2020
* 发表于：PAKDD 2020
* 标签：DTDG，图神经网络
* 概述：目前的方法使用了时态约束权重（temporal regularized weights）来使节点在相邻时态状态的变化是平滑的，但是这种约束权重是不变的，无法反映图中节点随时间演化的规律。本文借鉴了GAT的思路，提出了TCN。但作者提到本文的贡献只是提高了精度，感觉并不是很有说服力。
* 链接：https://link.springer.com/chapter/10.1007/978-3-030-47426-3_32
* 相关数据集：
    * Enron
    * UCI
    * Yelp
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

#### Continuous-Time Dynamic Graph Learning via Neural Interaction Processes
* 作者： Xiaofu Chang, et al.(Ant Group)
* 发表时间：2020
* 发表于：CIKM '20: Proceedings of the 29th ACM International Conference on Information & Knowledge Management
* 标签：CTDG，异质信息，时态点序列过程
* 概述：针对动态图中并存的拓扑信息与时态信息，本文提出了TDIG(Temporal Dependency Interaction Graph)的概念，并基于该概念提出了一种新的编码框架TDIG-MPNN，能够产生连续时间上的节点动态表示。该框架由TDIG-HGAN与TDIG-RGNN组成。前者能够聚合来自异质邻居节点的局部时态与结构信息；后者使用LSTM架构建模长序列的信息传递，整合了TDIG-HGAN的输出，捕捉全局的信息。此外，作者采用了一种基于注意力机制的选择算法，能够针对某一节点u，计算历史与其关联的节点对其不同重要程度分值。在训练过程中，作者将其定义为一个时态点序列过程(Temporal Point Process)问题进行优化。在实验中，作者针对时态链接预测问题，通过hit@10/Mean Rank指标对一些经典的静态图表示学习算法与STOA的动态图表示学习方法进行了对比，作者提出的模型在多个Transductive与一个Inductive数据集上取得了最好的效果。
* 链接：https://dl.acm.org/doi/pdf/10.1145/3340531.3411946
* 相关数据集：
    * CollegeMsg (Transductive)
    * Amazon (Transductive)
    * LastFM  (Transductive)
    * Huabei Trades (Inductive)
* 是否有开源代码：无

#### A Data-Driven Graph Generative Model for Temporal Interaction Networks
* 作者： Dawei Zhou, et al.(UIUC)
* 发表时间：2020
* 发表于：KDD 2020
* 标签：CTDG，图生成模型
* 概述：这篇论文是一篇深度图生成领域的文章，作者将动态图生成领域与transformer模型结合，设计了一种端到端的图生成模型TagGen。TagGen包含一种新颖的采样机制，能够捕捉到时态网络中的结构信息与时态信息。而且TagGen能够参数化双向自注意力机制，选择local operation，从而生成时态随机游走序列。最后，一个判别器（discriminator）在其中选择更贴近于真实数据的随机游走序列，将这些序列返回至一个组装模块（assembling module），生成新的随机游走序列。
作者在7个数据集上进行了实验，在跨度不同的指标中，TagGen表现更好；在具体任务（异常检测，链接预测）中，TagGen大幅度提升了性能。
* 链接：https://www.kdd.org/kdd2020/accepted-papers/view/a-data-driven-graph-generative-model-for-temporal-interaction-networks
* 相关数据集：
    * DBLP 
    * SO
    * MO
    * WIKI
    * EMAIL
    * MSG
    * BITCOIN
* 是否有开源代码：有 (https://github.com/davidchouzdw/TagGen )，但是其代码存在问题，如测试时未读取训练好的模型等。


#### Embedding Dynamic Attributed Networks by Modeling the Evolution Processes
* 作者： Zenan Xu, et al.
* 发表时间：2020
* 发表于：COLING 2020
* 标签：DTDG，Dynamic Attributed Networks
* 概述：作者提出了一种可以在动态属性网络进行表示学习的模型Dane，该模型可以在离散的属性动态图上进行表示学习工作。具体地，该模型包括Activeness-aware Neighborhood Embedding与Prediction of the Next-Timestamp Embedding两个模块。第一个模块提出了activeness-aware neighborhood embedding方法，利用了注意力机制，有权重地聚合邻居的不同特征；第二个模块也采用了注意力机制，避免了RNN等模型长距离遗忘的缺点，能够依据节点的历史SNAPSHOT状态学习到不同的权重。作者在动态链接预测与动态节点分类两个任务上进行了实验。
* 链接：https://www.aclweb.org/anthology/2020.coling-main.600/
* 相关数据集：
    * MOOC 
    * Brain
    * DBLP
    * ACM
* 是否有开源代码：无

#### Learning to Encode Evolutionary Knowledge for Automatic Commenting Long Novels
* 作者： Canxiang Yan, et al.
* 发表时间：2020
* 发表于：arXiv
* 标签：动态知识图谱，knowledge，DTDG
* 概述：长篇小说文本的自动评注任务（auto commenting task）需要依据小说文本中提及的人物，以及人物之间的关系，为小说文本自动生成自然语言表述的评注。小说中的人物及人物关系是动态演变的，静态知识图谱无法建模这种演变关系。基于此，作者设计了GraphNovel数据集，提出了演化知识图谱（Evolutionary Knowledge Graph）的框架，为每一章节的人物节点建立关系。给定一段需要评注的小说文本，框架能够整合文本中提及人物节点过去与未来的embedding，并通过一个graph-to-sequence模型生成评注文本。
* 链接：https://arxiv.org/abs/2004.09974
* 相关数据集：
    * GraphNovel
* 是否有开源代码：无


#### Generic Representation Learning for Dynamic Social Interaction
* 作者： Yanbang Wang, et al.
* 发表时间：2020
* 发表于：KDD
* 标签：时态网络，Dynamic Social Interaction
* 概述：社交互动（Social interactions）能够反应人类的社会地位与心理状态。社交关系是动态演变的，因此，在一个人群中，人们之间互相的动作能够反应这种模式。传统的方法一般适用人工定义模板的方法作者使用时态网络定义该问题，提出了一种temporal network-diffusion convolution network的方法，并在三个不同的数据集中对三种不同的心理状态进行了预测。
* 链接：http://www.mlgworkshop.org/2020/papers/MLG2020_paper_6.pdf
* 相关数据集：
    * RESISTANCE-1/2/3
    * ELEA
* 是否有开源代码：无

#### Motif-Preserving Temporal Network Embedding
* 作者： Hong Huang, et al.(hust)
* 发表时间：2020
* 发表于：IJCAI 2020
* 标签：CTDG，motif，hawkes
* 概述：本论文采用了一种meso-dynamics的建模方法，通过一种时序网络上的motif——open triad，考虑三个节点之间的triad结构，利用Hawkes过程建模节点对之间的密度函数，来学习时态网络中的embedding。论文在节点分类、链接预测（这一部分实验写的不清楚，不太明白是怎么做的实验）、链接推荐上取得了较好的效果。
* 链接：https://www.ijcai.org/Proceedings/2020/0172.pdf
* 相关数据集：
    * School 
    * Digg
    * Mobile
    * dblp
* 是否有开源代码：无

#### Motif-Preserving Temporal Network Embedding
* 作者： Hong Huang, et al.(hust)
* 发表时间：2020
* 发表于：IJCAI 2020
* 标签：CTDG，motif，hawkes
* 概述：本论文采用了一种meso-dynamics的建模方法，通过一种时序网络上的motif——open triad，考虑三个节点之间的triad结构，利用Hawkes过程建模节点对之间的密度函数，来学习时态网络中的embedding。论文在节点分类、链接预测（这一部分实验写的不清楚，不太明白是怎么做的实验）、链接推荐上取得了较好的效果。
* 链接：https://www.ijcai.org/Proceedings/2020/0172.pdf
* 相关数据集：
    * School 
    * Digg
    * Mobile
    * dblp
* 是否有开源代码：无


#### Local Motif Clustering on Time-Evolving Graphs
* 作者： Dongqi Fu, et al.(UIUC)
* 发表时间：2020
* 发表于：KDD 2020
* 标签：DTDG，motif，cluster
* 概述：图的motif是研究复杂网络的一种手段，能够揭示图形成的规律。motif clustering通过挖掘图中存在motif的不同形式，寻找图中节点的聚类簇。目前，局部聚类技术（一种聚焦于一组种子节点并为其划分cluster）已经广泛应用于静态图中，但在动态图领域尚未被应用。基于此，作者提出了一种适用于时态演化图（time- evolving graph）的局部motif聚类算法（L-MEGA）。在该算法中，作者设计了edge filtering/motif push operation与incremental sweep cut等技术，提高了算法的性能和效率。
* 链接：https://dl.acm.org/doi/10.1145/3308560.3316581
* 相关数据集：
    * Alpha
    * OTC
    * Call
    * Contact
* 是否有开源代码：有（https://github.com/DongqiFu/L-MEGA ）



#### INDUCTIVE REPRESENTATION LEARNING ON TEMPORAL GRAPHS
* 作者： Da Xu, et al.
* 发表时间：2020
* 发表于：ICLR 2020
* 标签：CTDG，inductive learning
* 概述：传统动态图表示学习的工作是transductive的，意即只能对训练集中出现过的节点进行表示，无法对unseen nodes进行表示。作者受到静态图中GraphSage、GAT等inductive learning方法的启发，提出了temporal graph attention layer（TGAT）这一结构。该结构使用了通过Bochner定理推导出时态核函数的时态编码模块，建模节点embedding识别为时间的函数，并能够随着图的演化，来有效聚合时态-拓扑邻居特征，从而学习到节点的时态-拓扑邻居聚合函数，使用inductive的方法快速生成节点表示。
* 链接：https://arxiv.org/abs/2002.07962
* 相关数据集：
    * Wikipedia
    * Reddit
    * Industrial dataset
* 是否有开源代码：无


#### INDUCTIVE REPRESENTATION LEARNING IN TEMPORAL NETWORKS VIA CAUSAL ANONYMOUS WALKS
* 作者： Yanbang Wang, et al.(stanford snap团队)
* 发表时间：2021
* 发表于：ICLR 2021
* 标签：CTDG，inductive learning，causal anonymous walk
* 概述：时态网络的演化是存在一定规律的，如社交网络中存在广泛的三元组闭环规律。作者认为，时态图上的inductive算法应该能够学习到这种规律，并应用至训练阶段未见过的数据中。为了表征这种规律，过滤掉节点特征对学习这种规律的影响，作者提出了基于Causal Anonymous Walks的节点表征方式，能够匿名化采样时态因果路径上的节点信息，从而对采样到的motif进行真正的关注学习。
* 链接：https://arxiv.org/abs/2101.05974
* 相关数据集：
    * Wikipedia
    * Reddit
    * MOOC
    * Social Evolution
    * Enron
    * UCI
* 是否有开源代码：有（https://github.com/snap-stanford/CAW ）

## Related Datasets 
包含一些知名的动态网络数据集，以及能够下载动态网络数据集合的网站。

动态网络数据集（仅展示部分）：
* Social Evolution Dataset
* Github Dataset
* GDELT (Global data on events, location, and tone)
* ICEWS (Integrated Crisis Early Warning System)
* DBLP：https://dblp.uni-trier.de/xml/
* FB-FORUM
* UCI Message data
* YELP：https://www.yelp.com/dataset/download
* MovieLens-10M

动态网络数据集合网站：
* SNAP数据集合网站：http://snap.stanford.edu/data/index.html
* SNAP时态数据集合：http://snap.stanford.edu/data/index.html#temporal
* KONECT数据集合网站（部分数据集的edge带有时间戳，可看作时序数据）  http://konect.cc/
* LINQS数据集合网站 https://linqs.soe.ucsc.edu/data
* CNets数据集合网站 https://cnets.indiana.edu/data-repository-for-nan-group/
* Network Repository（包含了数千个网络数据集，且包含简单的可视化与数据集统计）：http://networkrepository.com/
* Network Repository中的动态网络数据集：http://networkrepository.com/dynamic.php
* Aminer学术社交网络数据集：https://www.aminer.cn/data
* 社会行为模式数据集：http://www.sociopatterns.org/datasets/

## 其他参考资料


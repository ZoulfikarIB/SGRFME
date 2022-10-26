# Code for reproduction of the paper:
Joint Graph and Reduced Flexible Manifold Embedding for Scalable Semi-Supervised Learning.
# Abstract
Recently, graph-based semi-supervised learning (GSSL) has received much attention. On the other hand, less attention has been paid to the problem of large-scale GSSL for multi-class classification.Existing scalable GSSL methods rely on a hard linear constraint. They cannot predict the labelling of test samples, or use predefined graphs, which limits their applications and performance. In this paper, we propose an algorithm that can handle large databases by using anchors. The main contribution compared to existing scalable semi-supervised models is the integration of the anchor graph computation  into the learned model. We develop a criterion to jointly estimate the unlabeled sample labels, the mapping of the feature space to the label space, and the affinity matrix of the anchor graph. Furthermore, the fusion of labels and features of anchors is used to construct the graph. Using the projection matrix, it can also predict the labels of the test samples by linear transformation. Experimental results on the large datasets NORB, RCV1 and Covtype show the effectiveness, scalability and superiority of the proposed method.

Keywords: Scalable semi-supervised learning, Reduced Flexible Manifold Embedding, Graph structured data, Inductive semi-supervised models.
# Our test is on:
PC with i9-7960@2.80 GHZ CPU <br/>
125 GB RAM <br/>
Matlab version R2018a. <br/>
# Databases:
All the data used in our paper can be found on:  <br/>
1- https://cs.nyu.edu/ylclab/data/norb-v1.0-small/ <br/>
2- http://archive.ics.uci.edu/ml/datasets/Covertype
# Contact
Please feel free to email me (zoulfikaribrahim32@gmail.com) for any questions about this work.

# Visual Boundary Detection: A Deep Neural Prediction Network and Quality Dissection

## Informal Summary

## Highlights

### Abstract
Notable aspects of the work:
1. Use of _covariance features_ which depend on _squared_ response of a filter to the input image
2. Integration of image information from multiple scales and semantic levels via multiple streams of interlinked, layered, and non-linear deep processing


### Introduction
_Boundary_ definition from Martin et al. [2004]:

>  **A boundary is a contour in the image plane that represents a change in pixel ownership from one object or surface to another.**

Boundary detection is involved with detecting abrupt changes in more global properties, such as texture, and therefore needs to integrate information across the image.

_Edge Detection_:
> Edge detection, is a low-level technique to detect an abrupt change in some image feature such as brightness or color.

### Methods for Visual Boundary Prediction
Two part architecture:
1. First part performs feature extraction, using unsupervised feature learning
2. Second part uses the features for boundary prediction, by using a variant of the mean-and-covariance RBM (mcRBM) [Ranzato and Hinton, 2010] architecture and its deep belief net extension.


#### Unsupervised feature learning
Is used for pretraining. A mcRBM which is a generative model for images is used. Model architecture isn't trivial, I didn't understand it completely.


#### Supervised boundary prediction
A feedforward sigmoidal neural network is used for boundary detection.

Three architectures are considered:
1. **_Shallow network_**: only a single hidden layer is used.
2. **_Deep Stream_**: uses mcDBN-type hidden units.
3. **_Two-Stream_**: uses the connection patterns of both former architectures via _skip-layer_ connections.

Each stream has two parts:
1. Image feature extraction
2. Hypothesis propagation/read-out

Multiple streams are motivated by the need to capture image information at multiple scales and semantic levels.

#### Related work
**gPb** => "globalization" step over a probability of boundary based on spectral clustering

The structured decision tree framework of [Dollar and Zitnick 2013] **learns the leaf labels as part of the tree training**.


### Methods for Visual Boundary Prediction Quality Assessment
**Very good discussion on quality assessment**

> A `P-R` curve can be summarized by computing the maximal `F-measure` score out of the points corresponding to thresholding with a particular value. This threshold can either be _optimized across the data set_ (`ODS`), or _on a per image basis_ (`OIS`). The F-score (`ODS`) is considered the _main metric of the benchmark_.

**consensus boundaries**: boundaries which all annotators tend to agree on. They make up 30.58% of all annotated boundaries.

**orphan boundaries**: _weak_ boundaries, which are marked only by a few, or even just a single annotator. They make up 30.15% of all annotated boundaries.


**Precision bubble**: Benchmarking protocol tends to reward algorithms for focusing on the orphans. This is the name of the phenomenon.

Another problem with evaluation protocol is that pixel-wise independent computation of hits and misses does not necessarily capture the _perceptual_ quality of a boundary prediction (it ignores spatial coherence of a prediction).


The **MSSIM** scores generally agree very well with the (admittedly subjective) perceptual quality of such predictions.

### Visual Boundary Prediction Experiments

#### Training of the models

**Generative model** trained using stochastic gradient ascent for approximate maximum likelihood.

**Discriminative model** trained using SGA for maximizing the conditional likelihood of a ground-truth contour map **y** given an image.

**Enhancements**:
1. Standardizing image to have zero mean and standard deviation one. Makes network robust to shift and changes in scale of global intensity.
2. Averaging multuple ground truth boundary maps in training phase to give a single probability map **y**.
3. ...
4. Applying network to 16 rotated versions of image.
5. Applying non-maximum suppression by the Canny method to the network predictions.

#### Results and method comparison

| Method Name               | Prediction Time (s) |
| ------------------------- | ------------------- |
| global gPb                | 240                 |
| global SCG                | 280                 |
| local gPb                 | 60                  |
| local SCG                 | 100                 |
| DeepNet two-stream        | 0.1~0.2             |
| Sketch Tokens             | 1                   |
| Dollar and Zitnick (2013) | 1/6                 |

#### Dissecting the deep neural prediction network

**Shallow vs. deep networks**: Deep networks perform better than shallow networks. Shallow networks have very local receptive fields.

**Covariance units**: Networks with covariance units perform better than networks with mean units only.

**Unsupervised pre-training and fine tuning**: Positive effect of unsupervised pre-training increases when moving from shallow to deep networks.

### Discussion
> We finally observe significant benefits of generative pre-training, possibly indicating that the amount of annotated training data is limited relative to the difficulty of the task.

An interesting topic would be to train a generative model on joint distribution of the image and the contour data. Has applications in image prediction from boundary data (de-sketching) and image completion.

## Cite as
Kivinen, Jyri J., Christopher KI Williams, Nicolas Heess, and DeepMind Technologies. "Visual boundary prediction: A deep neural prediction network and quality dissection." In AISTATS, vol. 1, no. 2, p. 9. 2014.


## References
[Dollar and Zitnick 2013] Dollár, Piotr, and C. Lawrence Zitnick. "Structured forests for fast edge detection." In Computer Vision (ICCV), 2013 IEEE International Conference on, pp. 1841-1848. IEEE, 2013.

[Ranzato and Hinton, 2010] Ranzato, M., and Geoffrey E. Hinton. "Modeling pixel means and covariances using factorized third-order Boltzmann machines." In Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on, pp. 2551-2558. IEEE, 2010.

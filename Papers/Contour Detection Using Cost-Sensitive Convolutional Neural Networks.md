# Contour Detection Using Cost-Sensitive Convolutional Neural Networks

## Informal Summary
In this paper, they have taken a pre-trained model of AlexNet, chopped off the two fully connected layers and have upsampled the convolutional layers, have stacked convolutional layers on top of each other and have used the corresponding feature vector per pixel as their descriptor. One substantial contribution might be their fine-tuning of the network parameters. Though all in all, nothing stellar is present in this paper. BTW, they apply a non-maximal suppression as a post process on their network's output to thin edges.

## Highlights
### Abstract

### Introduction
Contour detection can be used for
* Image segmentation (Arbelaez et al., 2011 -> Contour detection and hierarchical image segmentation; Malik et al., 2001)
* Object detection (Zitnick & Dollar, 2014 -> Edge Boxes)
* Object Recognition (Shotton et al., 2008 -> Mutliscale categorical object recognition using contour fragments)

In our method, we intend to use the CNN architecture to generate features for each image pixel, not just a single feature vector for the whole input image.

### Related Work

#### Contour Detection
Textures are also useful local cues to increase the detection accuracy.

#### Convolutional Neural Networks

### Per-Pixel CNN Features
**DenseNet** provides fast mutliscale feature pyramid extraction af any Caffe convolutional neural network. DenseNet is an open source system that computes dense and multiscale features from the convolutional layers of a Caffe CNN based object classifier. Given an input image, DenseNet computes its multiscale versions and stitches them to a large plane. After processing the whole plane by CNNs, DenseNet would unstitch the descriptor planes and then obtain multiresolution CNN descriptor.
* Jua et al., Caffe: Convolutional architecture for fast feature embedding

#### DenseNet feature pyramids
Convolutional layers
1. Conv1 `(96 * 1)`
2. Conv2 `(256 * 1)`
3. Conv3 `(384 * 1)`
4. Conv4 `(384 * 1)`
5. Conv5 `(256 * 1)`

We train the SVM based on only the original resolution. In test time, we classify test images using both the original and the double resolutions. We average the two resulting edge maps for the final output of contour detection.

#### Per-Pixel Finetuning
The new  image (patch) size is set to `163 * 163`, and would reduce to `1 * 1` in Conv5, at which the 2-way softmax layer can now properly compute the per-pixel probability of being a contour point.

Note that the loss for backpropagation is computed by the label prediction and the ground truth of the center pixel of `163 * 163` patch.

#### Cost-Sensitive Fine-tuning
We set $\alpha = 2\beta$ for _positive_ cost-sensitive fine-tuning and $2\alpha = \beta$ for _negative_ cost-sensitive fine-tuning.

... that is for positive cost-sensitive fine-tuning, we sample twice more edge patches than non-edge ones.

#### Final Fusion Model

### Experiment Results
Prior to evaluation, we apply a standard non-maximal suppression technique to the edge maps to obtain thinned edges.

#### Experiments on Fine-tuning
In conclusion, per-pixel fine-tuning raises the performances of per-pixel applications.

#### Contour Detection Results and Comparisons

### Discussion
An interesting future research direction is to establish a proper **dimensionality reduction** framework for the resulting high-dimensional per-pixel feature vectors and to examine its effects on the performance of contour detection.

# Deep-Restore-PyTorch
Deep CNN for learning image restoration without clean data! (Noise2Noise)

## Introduction
Image restoration is task in which we have a noisy input image and we desire to get a noise free output image. Several techniques have been proposed for this task. One is using the Light Transport Simulation algorithm, which traces the path of millions of light rays. The disadvantage of this technique is the rendering time. It may take upto hours to render a single scene.</br>
Recently Convolutional Neural Networks have been proposed to solve this problem, which they do and also achieve state of the art results. They can learn the concept of noise, hence when provided with unseen noisy image, it can generate a noise free image easily in very less time. We can simply train a CNN for this task by providing noisy image as input and clean image as target and minimize a loss function using gradient descent.

![](http://zinggadget.com/wp-content/uploads/2018/07/Noise2Noise-Nvidia-Kecerdasan-Buatan.jpg)

## The Problem
It works. Works pretty well indeed. But there are cases where it is not just expensive but impossible to create clean images for our training data. Low light photography, Astronomical Imaging or Magnetic Resonance Imaging (MRI) are few of such cases. Neural Network based techniques cannot be easily used for this case.

## The Solution
Here comes a novel technique to address this issue. This method uses only noisy images to train the neural network to produce clean image as output. No clean images are required whatsoever for this technique.

## A Few Samples
### Multiplicative Bernoulli Noise Removal
![](imgs/index8.png) ![](imgs/index7.png)
### Gaussian Noise Removal
![](imgs/index4.png) ![](imgs/index3.png)
### Corrupt Text Removal
![](imgs/index2.png) ![](imgs/index.png)</br>
### Poisson Noise Removal
![](imgs/index6.png) ![](imgs/index5.png)
#### Note: I am not yet sure about how poisson noise needs to be implemented because it's not additive and is signal dependent while the paper simply mentions using a constant lambda.</br>

#### The minor artifacts seen are due to the following
1. Only 291 images were used for training.
2. The images were random cropped to 64 x 64 for quick training.</br>
Progressive resizing could be incorporated along with more images to get high resolution results.


## References:
[Noise2Noise Paper](https://arxiv.org/pdf/1803.04189.pdf)


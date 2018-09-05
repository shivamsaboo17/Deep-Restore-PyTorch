# Noise2Noise-Pytorch
Implementation of NVDIA's Noise2Noise paper in PyTorch for unsupervised denoising and text removal.

## Introduction
Image restoration is task in which we have a noisy input image and we desire to get a noise free output image. Several techniques have been proposed for this task. One is using the Light Transport Simulation algorithm, which traces the path of millions of light rays. The disadvantage of this technique is the rendering time. It may take upto hours to render a single scene.</br>
Recently Convolutional Neural Networks have been proposed to solve this problem, which they do and also achieve state of the art results. They can learn the concept of noise, hence when provided with unseen noisy image, it can generate a noise free image easily in very less time. We can simply train a CNN for this task by providing noisy image as input and clean image as target and minimize a loss function using gradient descent.

![](http://zinggadget.com/wp-content/uploads/2018/07/Noise2Noise-Nvidia-Kecerdasan-Buatan.jpg)

## The Problem
It works. Works pretty well indeed. But there are cases where it is not just expensive but impossible to create clean images for our training data. Low light photography, Astronomical Imaging or Magnetic Resonance Imaging (MRI) are few of such cases. Neural Network based techniques cannot be easily used for this case.

## The Solution
Here comes the amazing technique to address this issue. This method uses only noisy images to train the neural network to produce clean image as output. No clean images are required whatsoever for this technique.




 

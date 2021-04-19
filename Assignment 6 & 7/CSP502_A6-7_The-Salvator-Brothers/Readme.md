# Neural Style Transfer

This work presents a neural algorithm of artistic style, known as Style Transfer, that can separate and recombine the image content and style of natural images. The algorithm allows us to produce new images of high perceptual quality that combine the content of an arbitrary photograph with the appearance of numerous well-known artworks. This work provides insights into the deep image representations learned by different Convolutional Neural Network architectures viz. VGG- 16, VGG-19, and ResNet50.

## Table of Content

- [Introduction](#introduction)
- [Approach](#approach)
  - [Model Description](#model)
  - [Representations](#representations)
  - [Loss Functions](#loss-functions)
- [Results](#results)
- [Installation Guide](#installation-guide)
- [References](#references)
- [Contribution](#contribution)


## Introduction

Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

The principle of neural style transfer is to define two distance functions, one that describes how different the content of two images are, Lcontent, and one that describes the difference between the two images in terms of their style, Lstyle. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image.

## Approach



### Representations

In order to get both the content and style representations, we looked at some intermediate layers within our model. Intermediate layers represent feature maps that become increasingly higher ordered as we go deeper. We are used 
the network architecture of VGG16 and VGG19 pretrained image classification network. These intermediate layers are necessary to define the representation of content and style of our images. For an input image, we tried to match the corresponding style and content target representations at these intermediate layers.

### Model

#### VGG16

#### VGG19

#### ResNet50


### Loss Functions

We have defined 2 loss functions: **content** loss function and **style** loss function. The content loss function ensures that the activations of the high layers are similar between the content image and the generated image. The style loss function makes sure that the
correlation of activations in all the layers are similar between the style image and the generated image.

### Process Flow

![alt_text](https://github.com/jimil749/CSP520-Computer-Vision/blob/jimil/assignment-6/Assignment%206%20%26%207/Samaritans_6/images/nst_architecture.jpg)

## Results

We took 4 content and 5 style images and tried them over VGG16, VGG19 and ResNet50 models.

<table>
  <tr>
    <td></td>
    <td align="center">VGG16</td>
    <td align="center">VGG19</td>
    <td align="center">ResNet50</td>
  </tr>
  <tr>
    <td align="center" width=175> Content = New York, Style = Starry Night </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/1_NewYork_StarryNight_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/2_NewYork_StarryNight_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/3_NewYork_StarryNight_ResNet50.jpg" width=250 height=250></td>
  </tr>
 
  <tr>
    <td align="center" width=175 > Content = New York, Style = Scream </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/4_NewYork_Scream_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/5_NewYork_Scream_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/6_NewYork_Scream_ResNet50.jpg" width=250 height=250></td>
  </tr>
  
  
  <tr>
    <td align="center"  width=175> Content = New York (Low Resolution 128), Style = Scream </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/14_NewYork128LowRes_Scream_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/15_NewYork128LowRes_Scream_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/16_NewYork128LowRes_Scream_ResNet50.jpg" width=250 height=250></td>
  </tr>
  
  <tr>
    <td align="center"  width=175> Content = New York, Style = HoneyComb </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/7_NewYork_Honeycomb_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/8_NewYork_Honeycomb_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/9_NewYork_Honeycomb_ResNet50.jpg" width=250 height=250></td>
  </tr>
  
  <tr>
    <td align="center"  width=175> Content = New York, Style = Starry Night </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/10_GrandCanyon_StarryNight_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/11_GrandCanyon_StarryNight_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/12_GrandCanyon_StarryNight_ResNet50.jpg" width=250 height=250 alt="GrandCanyon x StarryNight : ResNet50"></td>
  </tr>
  
  
  <tr>
    <td align="center"  width=175> Content = New York, Style = Scream </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/12_GrandCanyon_Scream_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/13_GrandCanyon_Scream_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/main/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/14_GrandCanyon_Scream_ResNet50.jpg" width=250 height=250 alt="GrandCanyon x Scream : ResNet50"></td>
  </tr>
 </table>


## Platform 

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)

## Installation Guide

- Clone this repository using 
```
$ git clone https://github.com/Engineer1999/CSP520-Computer-Vision.git
$ cd CSP520-Computer-Vision
```
- Install the dependencies using
```
$ pip install -r requirements.txt
```
- To run locally, launch jupyter notebook using `$ jupyter notebook` or upload the `.ipynb` file on Google Colab.


## References
-  L. A. Gatys, A. S. Ecker, and M. Bethge, “A Neural Algorithm of Artistic Style,” arXiv.org, 02-Sep-2015. [Online]. Available: [Link](https://arxiv.org/abs/508.06576). 
- S. Desai, “Neural Artistic Style Transfer: A Comprehensive Look,” Medium, 14-Sep-2017. [Online]. Available: [Link](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199).
- TensorFlow, “Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution,” Medium, 27-Sep-2018. [Online]. Available: [Link](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
- N. Kasten, “Art & AI: The Logic Behind Deep Learning ’Style Transfer’,” Medium, 12-Mar-2020. [Online]. Available: [Link](https://medium.com/codait/art-ai-the-logic-behind-deep-learning-style-transfer-1f59f51441d1).
- Leon A. Gatys, ”Image Style Transfer Using Convolutional Neural Networks”, . [Online]. Available: [Link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).



## Contribution

- Team : The Salvator Brothers
- Members : [Kirtan Kalaria](https://github.com/kkalaria16), [Manav Vagrecha](https://github.com/caped-crusader16) 

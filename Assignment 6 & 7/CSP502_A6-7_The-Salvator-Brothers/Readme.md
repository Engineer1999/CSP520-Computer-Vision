# Neural Style Transfer

This work presents a neural algorithm of artistic style, known as Style Transfer, that can separate and recombine the image content and style of natural images. The algorithm allows us to produce new images of high perceptual quality that combine the content of an arbitrary photograph with the appearance of numerous well-known artworks. This work provides insights into the deep image representations learned by different Convolutional Neural Network architectures viz. VGG- 16, VGG-19, and ResNet50.

## Table of Content

- [Introduction](#introduction)
- [Approach](#approach)
  - [Models Used](#models)
    - [VGG16](#1-vgg16)
    - [VGG19](#2-vgg19)
    - [ResNet50](#3-resnet50)
  - [Representations and Loss Functions](#representations-and-loss-functions)
- [Information Regarding Dataset](#dataset-information)
- [Results](#results)
- [Discussion Over Results](#discussion-over-results)
  - [A.) Effect of model architecture](#a-effect-of-model-architecture)
  - [B.) Effect of content and style loss weights](#b-effect-of-content-and-style-loss-weights)
  - [C.) Effect of content and style image type](#c-effect-of-content-and-style-image-type)
  - [D.) Effect of content image resolution](#d-effect-of-content-image-resolution)
- [Key-Takeaways](#key-takeaways)
- [Installation Guide](#installation-guide)
- [References](#references)
- [Contribution](#contribution)

* * *

## Introduction

Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

The principle of neural style transfer is to define two distance functions, one that describes how different the content of two images are, Lcontent, and one that describes the difference between the two images in terms of their style, Lstyle. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image.

* * *

## Approach

### Models

In this work, we have used 3 well-known CNN architectures and compared their resulting images.

#### 1.) VGG16

![VGG16](https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/vgg16.png)

#### 2.) VGG19

![Vgg19](https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/vgg19.jpg)

#### 3.) ResNet50

![ResNet50](https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/resnet50.png)

### Representations and Loss Functions

In order to get both the content and style representations, we looked at some intermediate layers within our model. Intermediate layers represent feature maps that become increasingly higher ordered as we go deeper. We are used the network architecture of VGG16 and VGG19 pretrained image classification network. These intermediate layers are necessary to define the representation of content and style of our images. For an input image, we tried to match the corresponding style and content target representations at these intermediate layers.

As the output image is framed from 2 images, there are basically, 2 loss functions: 
* *Content Loss function (Lcontent)* - It ensures that the activations of the high layers are similar between the content image and the generated image.
 
* *Style Loss function (Lstyle)*  It ensures that the correlation of activations in all the layers are similar between the style image and the generated image.

<img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/image1.png" width=500 height=500>

<img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/image2.png" width=500 height=500>

### Process Flow

![Process Flow](https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Process_Flow.png)

* * *

## Dataset Information

You may take any images as content image and style image as per out wish. We took 4 content and 5 style images which are attached in the [`content`](https://github.com/caped-crusader16/CSP520-Computer-Vision/tree/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Content%20Images) and [`style`](https://github.com/caped-crusader16/CSP520-Computer-Vision/tree/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Style%20Images) folders in image section.

* * *

## Results

We tried each an every possible combination of Content and Style images over VGG16, VGG19 and ResNet50 models. The following table constains few of those images to show clear comparison.

<table>
  <tr>
    <td></td>
    <td align="center">VGG16</td>
    <td align="center">VGG19</td>
    <td align="center">ResNet50</td>
  </tr>
  <tr>
    <td align="center" width=175> Content = New York, Style = Starry Night </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/1_NewYork_StarryNight_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/2_NewYork_StarryNight_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/3_NewYork_StarryNight_ResNet50.jpg" width=250 height=250></td>
  </tr>
 
  <tr>
    <td align="center" width=175 > Content = New York, Style = Scream </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/4_NewYork_Scream_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/5_NewYork_Scream_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/6_NewYork_Scream_ResNet50.jpg" width=250 height=250></td>
  </tr>
  
  
  <tr>
    <td align="center"  width=175> Content = New York (Low Resolution 128), Style = Scream </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/14_NewYork128LowRes_Scream_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/15_NewYork128LowRes_Scream_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/16_NewYork128LowRes_Scream_ResNet50.jpg" width=250 height=250></td>
  </tr>
  
  <tr>
    <td align="center"  width=175> Content = Grand Canyon, Style = Starry Night </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/10_GrandCanyon_StarryNight_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/11_GrandCanyon_StarryNight_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/GrandCanyon-X-StarryNights-ResNet50.png" width=250 height=250 alt="GrandCanyon x StarryNight : ResNet50"></td>
  </tr>
  
  
  <tr>
    <td align="center"  width=175> Content = Grand Canyon, Style = Scream </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/12_GrandCanyon_Scream_VGG16.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/13_GrandCanyon_Scream_VGG19.jpg" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting Outputs/GrandCanyon-X-Scream-ResNet50.png" width=250 height=250 alt="GrandCanyon x Scream : ResNet50"></td>
  </tr>
  
  <tr>
    <td align="center"  width=175> Content = Despicable Me, Style = Honey Comb </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting%20Outputs/DespicableMe-X-HoneyComb-VGG16.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting%20Outputs/DespicableMe-X-HoneyComb-VGG19.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A6%267/Assignment%206%20%26%207/CSP502_A6-7_The-Salvator-Brothers/images/Resulting%20Outputs/DespicableMe-X-HoneyComb-ResNet50.png" width=250 height=250 alt="Despicable Me x Honey Comb : ResNet50"></td>
  </tr>
 </table>

Many More images have been generated. Please check those in the [ipynb notebook](https://colab.research.google.com/drive/1tBeE7PvA_8isnkHtPkSe4fL4BbWu1Nf2?usp=sharing).

* * *

## Discussion Over Results

Neural style transfer is more art than science. The quality of result in style transfer, is largely a subjective matter. The expectations can vary from person to person, so does the likeability of the result. Much also depends on the difference between the expectation and the result and the flexibility of the viewer. So, we will discuss the following according to our point of view.

### A) Effect of model architecture

VGG16 and VGG19 have much better performance than ResNet50. VGG models succeed in capturing the style much better than ResNet50, even though they were trained on the same ImageNet dataset. Although VGG16 and VGG19 perform similarly, they have some noticeable differences. 

VGG19 produces relatively smoother images but slightly deforms the structure of objects. On the other hand, VGG16 captures exact style elements (type of higher complexity features) from the style image better, for example in Fig. 22, we can see the yellow spots in the blue strokes exactly like in the original style image, while VGG19 captures lower level
features like the strokes of alternating shades of blue better. This also contributes to the smoothness of the result image as lesser complex features are blended better in VGG19, but at the slight cost of exact stylistic elements. We can’t declare one better than the other; the more suitable model depends on the what type of stylistic elements (i.e. features of the style image) we want to capture, rendering it a subjective choice. 

Meanwhile, ResNet50, irrespective of the choice of style and content layers (which is an extremely cumbersome process), captures content much better but fails to capture complex style features satisfactorily. However, in its own way, the resulting image is not unpleasant and looks more like an oil painting. On the other hand, VGG images look like crayon or
water color paintings. 

It would be interesting to see the effect of using representation layers from VGG16 or VGG19 as well as ResNet50. From literature review, we even found some people who have tried these models pre-trained on datasets other than ImageNet. They observed that the results we get are not due to the
dataset used but due to the architecture of the models. In other datasets, they found the same differences in performance and results. It appears that because ResNet50 is very complex, the style features are distributed in a very large number of layers, reducing the variation covered in each block. Think of
it as ’feature density’. Hence, ResNet50, and by extension, all ResNet models fail to capture complex styles without significant loss in content. The difference in performance of the two VGG models can also be explained by the same logic. However, very low architecture complexity may result in the style and content not being captured well. Hence there is a sweet spot for model complexity and VGG16 and VGG19 lie close to it.

Also, as the model complexity increases, we have to include
more layers starting from the beginning to cover larger scale
style features, which results in other set of issues.

### B) Effect of content and style loss weights

We isolate style and content in the style transfer algorithm
but it is not possible to do it perfectly. At best, we create two
sets of representations - one with more content representation
than style representation and the other the other way around.
We rely on the difference in each representation to effective
represent the style and the content.
Hence, we cannot perfectly combine the style from the style
image and content from the content image. However, we can
make different choices of the representational layers to suit our
need. And after that we calculate the style loss and content
loss.
But in order to optimise the result, we need just one loss that
includes the style and content losses. We can tune the process
to balance the emphasis style and content by controlling the
weights of style and content losses in the total loss

Higher content loss weight will help preserve the visual
structure of the content image but will compromise the style
and vice versa for higher content loss weight. There is a clear
trade-off. The key is to find the right balance by adjusting the
ratio of the two.

### C) Effect of content and style image type

Natural images tend to be smoother than urban or artificial
images, including animated images. Moreover, the former has
a lesser geometric perfection than the latter, like sharp edges
and corners. Thus the loss of content is more prominent in the
latter.
Moreover, certain pairings of style and content images tend
to perform better. There is no concrete logic to explain this. It
just is, just like wine and cheese pairings. It is also observed
that natural content images perform better with natural style
images like artwork.

### D) Effect of content image resolution

It appears that low resolution content images result in
smoother result images. It has lesser noise like fewer mono-
color blobs and patches.However, content structures, especially
finer structures are also compromised marginally in case of
low resolution images.
The better choice depends on the type of style features we
are looking for in the style image. If we want more localized
features, high resolution content images are better.
We believe the same could be achieved with regularisation
layers in the models, especially blurring or smoothing layers.

* * *

## Key Takeaways

In this work, we have implemented neural algorithm for
Neural Style Transfer using few pre-trained classical CNN
architectures such as VGG16, VGG19 and ResNet50 and
their results have been discussed and compared. Conclusively,
this process is not suited for batch processing. Rather, it
is to be implemented as a customized process by choosing
representation layers and hyperparameters for content and
style image pairs to get the desired result. We can leverage
the inferences and learnings from the discussion of results
and use it to improve the process in many ways. Since it is
a relatively newer topic, very less work has been done in this
area but there is a large scope of improvement, which again,
lies in the interdisciplinary intersection of art and computer
vision, requiring sufficient proficiency in both fields.

* * *

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

* * *

## References
-  L. A. Gatys, A. S. Ecker, and M. Bethge, “A Neural Algorithm of Artistic Style,” arXiv.org, 02-Sep-2015. [Online]. Available: [Link](https://arxiv.org/abs/508.06576). 
- S. Desai, “Neural Artistic Style Transfer: A Comprehensive Look,” Medium, 14-Sep-2017. [Online]. Available: [Link](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199).
- TensorFlow, “Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution,” Medium, 27-Sep-2018. [Online]. Available: [Link](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
- N. Kasten, “Art & AI: The Logic Behind Deep Learning ’Style Transfer’,” Medium, 12-Mar-2020. [Online]. Available: [Link](https://medium.com/codait/art-ai-the-logic-behind-deep-learning-style-transfer-1f59f51441d1).
- Leon A. Gatys, ”Image Style Transfer Using Convolutional Neural Networks”, . [Online]. Available: [Link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

* * *

## Contribution

- Team : The Salvator Brothers
- Members : [Kirtan Kalaria](https://github.com/kkalaria16), [Manav Vagrecha](https://github.com/caped-crusader16)
      

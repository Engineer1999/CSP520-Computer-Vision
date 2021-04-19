 # Citrus Leaf Classification with CNN

In this assignment, our objective is Classification of Citrus Leaves Data using a CNN classifier. Here, we are comparing performances of different optimizers and hyper-parameters on the basis of different metrics like  Accuracy, Precision, Recall.

## Table of Content

- [Introduction](#introduction)
- [Approach](#approach)
 - [Dataset and preparation](#dataset-and-preparation)
 - [Model](#model)
 - [Training](#training)
  - [Optimizers](#optimizers)
  - [Hyperparameters](#hyperparameters)
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

Image classification is amongst the fundamental tasks handled by CNN. The goal in classification is to assign a label to an image. The classification comprehends the
image. In this assignment, the objective is to understand, design, and implement a CNN classifier. We must not just implement the CNN classifier but understand it as
well.

* * *

## Approach

### Dataset and preparation

The original dataset contains 759 images of healthy and un-healthy citrus fruits and leaves. However, as of now the owners only export 594 images of citrus leaves with the following 4 labels: Black Spot, Canker, Greening, and Healthy. The exported images are in PNG format and have the dimension 256×256.

ImageDataGenerator was used to generate training, validation and testing data [60%,20%,20%] from the dataset. This allowed us to randomly augment the training data by zooming in and out (30%), rotating (±180°), height and width shifting (30%), and horizontal and vertical flipping. Finally, both training and validation images were re-scaled pixel-wise to the intensity in range [0,1].

### Model

We made our custom model for this task.

#### Architecture

Layers:
- Convolution (16 filters, 3×3 kernel, ReLU)
- MaxPooling (/2)
- Convolution (32 filters, 3×3 kernel, ReLU)
- MaxPooling (/2)
- Convolution (64 filters, 3×3 kernel, ReLU)
- MaxPooling (/2)
- Convolution (64 filters, 3×3 kernel, ReLU)
- MaxPooling (/2)
- Convolution (64 filters, 3×3 kernel, ReLU)
- MaxPooling (/2)
- Flatten [4092 units]
- Dense (512, ReLU)
- Dense (4, Sigmoid)

![CNN Architecture](https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/Convnet.png)

#### Inspiration

The inspiration for our architecture is drawn from Alexnet. We have a simple dataset so it was important to control the number of  parameters in order to control variance. This particular model has been reached after more than a hundred tries and iterative steps in order to improve it. All tries at adding regularization, changing activation functions, ordering of layers, number of layers and more, have resulted in very unfavorable accuracies. This is a sweet spot we had reached.

### Training

#### Optimizers
We  have  tried  Stochastic  Gradient  Descent  (SGD)  withmomentum,  Adam  and  RMSprop.  SGD  performed  theworst and RMSprop performed the best.

##### Stochastic Gradient Descent, with momentum
![SGD](https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/SGD.jpg)

##### Root Mean Square Propogation
![RMSPROP](https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/RMSPROP.jpg)

##### Adaptive Moment Estimation
![ADAM](https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/ADAM.jpg)

#### Hyperparameters

- Number of Epochs
- Batch Size

We have kept the number of epochs as 35 as to neither underfit nor overfit the model, and the ratio of batch sizes of training and validation data (32 and 8) close to the ratio of their share in the dataset (3).

* * *

## Results

### Performance metrics
<table>
  <tr>
    <td></td>
    <td align="center">Loss</td>
    <td align="center">Accuracy Prop</td>
    <td align="center">Precision</td>
    <td align="center">Recall</td>
  </tr>
  <tr>
    <td align="center" width=175> SGD </td>
    <td>0.9554</td>
    <td>0.4832</td>
    <td>0.6321</td>
    <td>0.3408</td>
  </tr>
 
  <tr>
    <td align="center" width=175> RMS Prop </td>
    <td>0.5202</td>
    <td>0.7849</td>
    <td>0.8049</td>
    <td>0.7374</td>
  </tr>
  
  <tr>
    <td align="center" width=175> Adam </td>
    <td>0.4764</td>
    <td>0.8156</td>
    <td>0.4140</td>
    <td>0.9944</td>
  </tr>
</table>
  
### Learning curves

<table>
  <tr>
    <td></td>
    <td align="center">SGD</td>
    <td align="center">RMS Prop</td>
    <td align="center">Adam</td>
  </tr>
 
  <tr>
    <td align="center" width=175> Loss Curves </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/loss_sgd.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/loss_rms.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/loss_adam.png" width=250 height=250></td>
  </tr>
  <tr>
    <td align="center" width=175> Accuracy Curves </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/accuracy_sgd.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/accuracy_rms.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/accuracy_adam.png" width=250 height=250></td>
  </tr>
  <tr>
    <td align="center" width=175> Precision Curves </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/precision_sgd.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/precision_rms.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/precision_adam.png" width=250 height=250></td>
  </tr>
  <tr>
    <td align="center"  width=175> Recall Curves </td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/recall_sgd.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/recall_rms.png" width=250 height=250></td>
    <td><img src="https://github.com/caped-crusader16/CSP520-Computer-Vision/blob/The_Salvator_Brothers-A4%265/Assignment%204%20%26%205/CSP502_A4-5_The-Salvator-Brothers/images/recall_adam.png" width=250 height=250></td>
  </tr>
</table>

* * *

## Discussion of Results

We can see that the maximum accuracy reached by the two best optimizers is about 80%, which is not bad considering the small size of dataset. From our observation, a sufficiently low learning rate along with a large number of epochs result in the most effective validation and testing accuracy. Learning curves served as a guide to determine the proper learning rate. In case of RMSprop, we can see that the learning curves are quite 'jumpy' or fluctuating, however, overall, with increase in number of epochs, the validation loss decreases and validation accuracy increases, so we know that the model is learning. 

A possible solution was to further decrease the learning rate and use an optimizer that utilizes momentum. However, when that was tried, we found that it results in relatively poorer loss and accuracy and hence the model doesn't train as well as desired. On the other hand, the decrease in learning rate even by a factor of 5 resulted in the weights to get stuck at some local minimum. Therefore, even with the volatility in RMSprop, the testing metrics are satisfactory, hence it is a good choice and learning can be considered satisfactory.

Adam is the also a very good option and yields balanced metrics. It has accuracy close to RMSprop yet significantly higher precision and recall than the latter. Overall, we can say that Adam is the best choice of optimizer in this case.

* * *

## Key Takeaways

In this work, we have implemented classification of leaf diseases using a custom made Convolutional Neural Network Neural Style Transfer, using different optimizers for training. Their results have been discussed and compared. Here as shown in the table, Adam and RMSprop have performed almost the similar where Adam seems to be the best choice as all the metrics have good and balanced values. Our final testing accuracy is ~80%.

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
-  V. Fung, “An overview of resnet and its variants,” 17-Jul-2017. [Online]. Available: [Link](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035).
- Vgg16 - convolutional network for classification and detection,” 24-Feb-2021. [Online]. Available: [Link](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
- N. Kasten, “Art & AI: The Logic Behind Deep Learning ’Style Transfer’,” Medium, 12-Mar-2020. [Online]. Available: [Link](https://neurohive.io/en/popular-networks/vgg16/).
- Keras Conv2D: Working with CNN 2D Convolutions in Keras”. [Online]. Available: [Link](https://missinglink.ai/guides/keras/keras-conv2d-working-cnn-2d-convolutions-keras/).

* * *

## Contribution

- Team : The Salvator Brothers
- Members : [Manav Vagrecha](https://github.com/caped-crusader16), [Kirtan Kalaria](https://github.com/kkalaria16)

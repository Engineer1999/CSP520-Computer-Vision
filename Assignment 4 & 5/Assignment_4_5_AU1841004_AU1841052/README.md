
<h1 align = "center">
    Homework Assignment 3 : Image Classification  
</h1>

---

<h1>Table of Content</h1>

- [Introduction](#introduction)
- [Approach](#Approach)
- [Results](#Results)
- [Platform](#Platform)
- [Installation guidelines](#Installation-guidelines)
- [References](#References)
- [Contributors](#Contributors)

## Introduction
---    
The human eye is capable of recognising, localising the features and classifying the images according to the variations present in the image at a very fast pace. In the man-made machines or systems, the capability of classifying the images according to the variations present is very less. The use of CNN in these systems help in increasing the classification accuracy as it creates a feature map for the features present within an image and predicts the label of the image using the minute-detailed as well as general features.

## Approach
---
- Dataset<br/>
 <p align="center">
<img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/Dataset_images.JPG" width="300" height="300" style="vertical-align:middle;margin:50px 50px">
 </p>
The dataset is of caltech_birds 2010. The caltech-UCSD Birds 200(CUB-200) consists of 200 categories of bird images. There are total 6033 images with annotations containing bounding boxes and segmentation labels.

- Data Preprcoessing <br/>
As the images are of different pixel ratio; in order to have easy in computation as well as to feed in the same pixel images to the model so that none of the features remains left out, the image pixels is made same. This is done by normalising the image where in every image is divided by 255, to keep the image pixels between 0 to 255. 
In the dataset, the image size is of about 180 X 180 due to which the system experiences much load, so the image size is reduced to a power of 2, here 64 X 64. But still due to heavy load on system, system crashes and so the size is reduced to power of 2, here 32 X 32 so that it can fit the model well.This size of image is further used.

- Data Augmentation<br/>
For a single image, augmentation in form of:<br/>
    - Flipping the image left to right or vice versa<br/>
    - Adjusting the image brightness to 0.4<br/>
    - Adjusting the image brightness to 0.2<br/>
    - Cropping the image to 0.5<br/>
    - Rotating the image several times by 90 degree<br/>
After these many types of augmentation, for every image about 7 new images are generated. In total around 50,000 new images are generated and the dataset is expanded.

- Model Architecture
<table>
  <tr>
    <td>Architecture</td>
     <td>Model Summary</td>
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/CNN_architecture.JPG" width=300 height=300></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/model_summary.JPG" width=300 height=300></td>
  </tr>
 </table>
As shown in the above figure,
    - We have used stride length instead of pooling because there is more information loss in using the pooling function.<br/>
    - Kernel size have been taken in order of 3, 5 and 3 in order to fetch the local features first, then the global features and then the remaining features so none of the features left untouched.<br/>
    - ReLU is used because as compared to tanh, sigmoid functions as it provides more generalisation accuracy.<br/>
    - Number of kernels have been taken in order of 64, 128, 64 as in some research work there is description that generally the number of filters are increased in first to maintain the spatial dimensions and then decreased to capture the important minute features.<br/>
    - Batch Normalisation is used so that in less number of epochs it provides more acceleration to training process.<br/>
    - More number of dropouts in the CNN layers increase the chances of making the learning process slow as more number of neurons freezes. So, we have used dropout only in the final layer with minimal value so that neither it makes the learning slow nor leads to overfitting.<br/>
    - The Flatten layer is used to convert the 2D matrix into 1D vector form.<br/> 
    - The dense layer after the flatten layer is used to connect the vector column to the neuron for final classification.<br/>
    - For final classification, mainly softmax activation function is used as it provides us with the probability that an input image belongs to a certain class with certain probability.<br/>
    - The final dense layer shape is of 200 such that it can predict the label of the input image from 200 categories available in the dataset.<br/>


## Results
---
- Accuracy<br/>
The accuracy of the training dataset is about 99\% while the accuracy of the testing dataset is about 55\%. This accuracy is due to large number of classes(200) available for classification due to which it leads to high chances of misclassification which in turn which less accurate results.

- Entropy Loss Graph 
<p align="center">
<img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/model_accuracy_loss_vs_epoch.jpeg" width="300" height="300" style="vertical-align:middle;margin:0px 50px">
</p>
This shows the curve depicting the relation between the training and testing loss with the number of epochs as well as the relation between the training and testing accuracy with the number of epochs.
 
- Graphs
<table>
  <tr>
    <td>Learning Rate</td>
     <td>Momentum</td>
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/model_learning_rate.png" width=300 height=300></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/model_momentum.png" width=300 height=300></td>
  </tr>
 </table>
The model with learning rate value of 0.1 or momentum value of 0.9 provides better results as compared to that model with learning rate and momentum combination which was even better compared to using the adam optimiser. The reason for choosing this values is that if the value chosen is too high then the model will learn quiclky and predicts the results quickly leading to misclassification and if value chosen is too low then the model may stop learning and may not be able to predict the results.

## Platform
---
- Google Colab


## Installation guidelines
---
- To clone this repository
 ```
git clone https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika.git
 ```
- To install the requirements
```
pip install -r requirements.txt
```
- To install the dataset
```
pip install -q tfds-nightly tensorflow matplotlib
```

## References
---
<a id="1">[1]</a> [Dataset](https://www.tensorflow.org/datasets/overview)<br/>
<a id="2">[2]</a>  [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)<br/>
<a id="3">[3]</a> [The Complete Beginner's Guide to Deep Learning: Convolutional Neural Networks](https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb)<br/>
<a id="4">[4]</a> 
[F. Sultana, A. Sufian, and P. Dutta,
Advancements in Image Classification using Convolutional Neural Network,
2018 Fourth International Conference on Research in Computational Intelligence and Communication Networks (ICRCICN).](https://ieeexplore.ieee.org/document/8718718)<br/>
<a id="5">[5]</a> 
[Dan C. Cires ̧an, Ueli Meier, Jonathan Masci, Luca M. Gambardella, and Jurgen Schmidhuber, 
Flexible, High Performance Convolutional Neural Networks for Image Classification,
International Joint Conference on Artificial Intelligence.](https://dl.acm.org/doi/10.5555/2283516.2283603)<br/>
<a id="6">[6]</a> 
[C.-C. Jay Kuo,
Understanding Convolutional Neural Networks with A Mathematical Model.](https://www.sciencedirect.com/science/article/abs/pii/S1047320316302267)<br/>



## Contributors
---

| [Dipika Pawar](https://github.com/DipikaPawar12)                                                                                                            | [Aanshi Patwari](https://github.com/aanshi18) |                                                                                                          

# Assignment 9

## Introduction 

Image classification is a standard problem in Computer
Vision and its has got huge applications. It refers to extract
information from the images and classify them into their respective classes. There are many ways to accomplish this. One
may extract features from the image using feature descriptors
like SURF, SIFT, HOG or ORB and then use a classifier(like
SVM) in order to classify them.
The advantage of choosing a Neural Network for classification, over the other algorithms is that it learns all the features
on its own, it gives better accuracy and it produces the output
which is not limited to the provided input. Also, there is no
need to store the input in a database, instead it stores the input
in its network itself.

We worked on creating a CNN classifier for identifying types of diseases in beans. There are two types of diseases in beans, bean rust and angular leaf spot. We developed a deep convolutional network which is responsible for classification of the disease. Data is classified into 3 classes healthy, bean rust and angular leaf spot.
<br/>
<img src=b1.jpg width="250">
<img src=b2.jpg width="250">

## Approach

All images used are of 500x500x3 for all Train, Validation and
Test Set.
Data Link : https://www.tensorflow.org/datasets/catalog/beans
- Reducing image size to 350x350x3 for all the Train,
Validation and Testing set, considering the coloured image.
- Normalising the train, validation and test set.
- Forming the deep CNN model(as per the code).
- Passing the reduced and normalised Train data and
their corresponding labels for model fitting.
- Evaluating the accuracy and loss for training and
validation set.
- Finally evaluating the test accuracy and plotting all
the results.
<br/>

### Below is the cnn model we built:
<img src=model.png width="350">

## Results
<img src=1.PNG width="400">
<img src=2.PNG width="400">

## Installation Guidelines and Platform Details

Run in Google Colab
There is a beans.ipynb file in the code section. You can directly open this jupyter notebook file in google colab and run the code by making the changes in the dataset location.

## Run in Jupyter Notebook

- First create an environment using anaconda
- Then install tensorflow using the command pip install tensorflow.
- Open the given beans.ipynb file and change the data folder location as per selected.



## References

- [Dataset](https://www.tensorflow.org/datasets/catalog/beans)
- [TFDS,Tensorflow datasets](https://www.tensorflow.org/datasets/api_docs/python/tfds)

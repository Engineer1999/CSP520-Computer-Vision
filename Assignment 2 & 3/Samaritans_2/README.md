# Face Recognition Challenge

## Introduction

Face recognition is one of the most sought-after technologies in the field of machine learning. In the recent times, the use case for this technology have broadened from specific surveillance applications in government security systems to wider applications across multiple industries for tasks such as user authentication, consumer experience, health and advertising. 

Just like any other form of biometric identification, face recognition requires samples to be collected, identified, extracted with necessary features and stored for recognitions. The entire face recognition problem is divided into following major segments:

1. Dataset Collection
2. Feature Extraction
3. Classification

## Approach

As mentioned in the introduction, the approach is broadly divided into 3 parts: 

### Dataset Collection

In order to obtain a robust face recognition model, we used sklearn's lfw dataset which consists of a database of face photographs designed for studying the problem of unconstrained face recognition. The dataset contains more than 13,000 images of faces collected from the web. Each face has been labeled with the name of the person pictured. 1680 of the people pictured have two or more disctint photos in the dataset. The images are in grayscale (pixel value = 0-255)

### Feature Extraction

After collecting the dataset, the next important step in face recognition is feature extraction. Faces are high dimensionality data consisting of a number of pixels. High dimensionality is difficult to process. Hence we used Histogram of Oriented Gradients, which is a feature extractor to extract out essential information out of the image.

### HoG Extractor

We have taken an input image of 64x64 px and applied HoG on it, which calculates the gradients and the direction of those gradients. The gradients are calculated convolving the following image kernels over the input image.

```
gy = [-1 0 1]

gx = [-1]
     [ 0]
     [+1]
```

The final magnitude and direction is given by:
 - Magnitude: G = <img src="https://render.githubusercontent.com/render/math?math=\sqrt{g_{x}^{2} %2B g_{y}^{2}}">
 - Direction/Orientation: <img src="https://render.githubusercontent.com/render/math?math=\theta"> = <img src="https://render.githubusercontent.com/render/math?math=arctan \dfrac{g_{y}}{g_{x}}">

These gradients help in removing the extra information like color information which is not necessary as the objects in the images can be easily detected with their shapes and edges. After this the next step performed was dividing the image into 8x8 cells and then calculating histogram for each cell. This makes the histogram less sensitive to noise as we are taking a larger patch compared to individual pixels. 

## Classification

After extracting the necessary features, we then used the SVM Classification model to classify the images into corresponding labels. Before training the SVM model, the pre-processed images were split into training and testing pairs with 25% of the data for testing and remaining 75% for training the SVM model. Sklearn's SVM function was used to perform this task. In order to obtain the best estimator, we performed hyperparameter tuning using `GridSearchCV`. After training, the model was tested on testing data, giving us the accuracy of **91.69%**

## Architectural Flow



## Results

## Installation Guides

## References

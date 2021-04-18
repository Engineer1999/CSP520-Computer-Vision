# Face Recognition Challenge

## Introduction
In this computer vision era, the potential to recognize human faces is evidence of spectacular human intelligence. Face recognition is known as one of the most significant and auspicious applications of image analysis. Especially during the previous years, face recogonition has gained remarkable attention.Psychologists concluded that holistic and feature-based approaches are two different routes of face recognition. For building this application we used SIFT ( scale-invariant feature transform ) feture extraction algorithm. The whole process of face recognition is divided into following major section:

1. Datast Creation (gathering)
2. Feature Extraction using SIFT Algorithm
3. Classification using SVM (Support Vector Machine)

## Approach

As we mentioned in the Introduction, while creating a this project we mainly focused on 3 parts:

### Dataset Creation

With the aim of obtain robust face recognition model, we created our own image database using our webcame. we collected 5 persons images. each person have 30 images so we have total 150  images of person's faces.Each faces has been labeled with the name of the person.

### Feature Extraction using SIFT Algorithm

After gathering the dataset, we are moving forward to second important stage of face detection - Feature Extraction. Dataset conatins high dimensional images. so it is very difficult to process this data. so we extract features using SIFT algorithm





## Results
![Output1 (1)](https://user-images.githubusercontent.com/60286760/115150811-a8536980-a087-11eb-8db1-54aead06adfb.png)
![Output2 (1)](https://user-images.githubusercontent.com/60286760/115150851-c6b96500-a087-11eb-9feb-ec7c5ff02d72.png)
![Output4 (1)](https://user-images.githubusercontent.com/60286760/115150869-e2247000-a087-11eb-8aa4-8c7b0d1164c4.png)



## Installation guidelines and platform details
Whole Face detection was performed on Google Colab. If you want to setup  working environment locally, download the "Face_Recognition_Challenge.ipynb" and upload to this file into the jupyter notebook and run jupyter notebook to use this script. In this project we created our own dataset. If you want to use our face database then Mount your drive (In google Colab). Ask supan.s2@ahduni.edu.in OR bhumiti.g@ahduni.edu.in for access to our face database. or If you want to create your own database then Choose your face data directory path (to load a blank directory and create your own database). Our directory will be loaded at '/gdrive/MyDrive/BitCoders/Assignment2/'.



Platform used:
[Google Colab](https://colab.research.google.com/)

## References

1. https://en.wikipedia.org/wiki/Scale-invariantfeaturetransform
2. https://www.researchgate.net/publication/224114966_Face_recognition_using_SIFT_features
3. https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification
4. https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html




## Contributors
---
| [Bhumiti Gohel](https://github.com/bhumiti28) | [Supan Shah](https://github.com/Supan14) |

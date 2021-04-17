## **Introduction:** 
</br> Convolutional Neural Network has been developed so well when it comes to image classification. There has been some amazing applications of Image classification in computer vision and classifying the disease infected leaves and normal leaves is one of the important application of Image classification. By automatic and an early diagnosis of a disease and its severity, effective, and timely treatment can be taken in advance [4]. The proposed work presents the comparison of two different Convolutional Neural Network (CNN) architectures to classify diseases of the citrus leaf. We have created our own CNN with 3 Conv - 3 Max pooling and 2 Dense layers [3] and also have added Dropout layer to overcome overfitting faced. This work also comprises of expermentation with hyper-parameters such as learning rate, dropout percentage and activation function at output layer and has compared the performance.
</br>

## **Approach:**
Approach will consists of the architecture of CNN which is used and the experiments we have carried out based on values of hyper-parameters.
#### ***Architecture***
There has been 3 Convolutional and 3 max pooling layer used by stacking one behind other. After getting higher level features, flattening of the feature matrix has been done and passed through Dense layers. There has been one dropout layer present in between two Dense layers.
#### ***Expermients with hyper-parameters***
There has been ReLU activation used for all the layers except output layer, and for the output layer two different activation functions have been experimented i.e., sigmoid and softmax. Sigmoid activation function maps any values from −∞ to ∞, to 0-1 and gives the predictive analysis of classification. Softmax activation is also famous for its mapping of large values to range of 0 − 1 as it is a probabilistic function which gives proportion of similarity in probabilistic manner to different class and hence it is used at the outputlayer. </br>
A list of learning rates, starting with 10^-3 and increment by multiple of 10 till 1. We have got five different values. Mapping of alpha values with index: [1 : 0.0001, 2 : 0.001, 3 : 0.01, 4 : 0.1, 5 : 1]. </br>
When we talk about accuracy plots between training and validation accuracy, there can be overfitting and underfitting where we don’t want to stuck. In first scenerio where we can see that training accuracy is quite higher than validation and also the rate of increment in validation accuracy is not so encouraging and thus we can say that there might be problem of overfitting.</br>

## **Results:**
Results of experimenting with Sigmoid and Softmax activation function. 
</br> Sigmoid function </br>
![Sigmoid function](https://github.com/yashpatel301/Computer-Vision-Basics/blob/main/Citrus-leaves-Classification/Results/LOSS_SIGMOID.png)

</br> Softmax function </br>
![Softmax function](https://github.com/yashpatel301/Computer-Vision-Basics/blob/main/Citrus-leaves-Classification/Results/LOSS_SOFTMAX.png)

</br>Results of experimenting with different learning rates.
</br> Accuracy </br>
![Accuracy](https://github.com/yashpatel301/Computer-Vision-Basics/blob/main/Citrus-leaves-Classification/Results/LR_accuracy.png)

</br> Validation Accuracy </br>
![Validation Accuracy](https://github.com/yashpatel301/Computer-Vision-Basics/blob/main/Citrus-leaves-Classification/Results/LR_val_acc.png)

</br>Results of experimenting with Dropout percentage.
</br> Dropout 30% </br>
![Accuracy](https://github.com/yashpatel301/Computer-Vision-Basics/blob/main/Citrus-leaves-Classification/Results/Dropout_30.png)

</br> Dropout 50% </br>
![Validation Accuracy](https://github.com/yashpatel301/Computer-Vision-Basics/blob/main/Citrus-leaves-Classification/Results/Dropout_50.png)

## **Installation guidelines and platform details:**
This assigment is performed on Google colab only. All the libraries are mentioned at the beginning of the code and are imported without any installation needed colab and for anaconda environment and all the required libraries are mentioned in the requirements.txt. 

## **References:** 
1. https://keras.io/api/preprocessing/image/
2. https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
3. https://www.geeksforgeeks.org/python-image-classification-using-keras/
4. Singh, U., Chouhan, S., Jain, S. and Jain, S., 2021. Multilayer Convolution Neural Network for the Classification of Mango Leaves Infected by Anthracnose Disease. [online] Ieeexplore.ieee.org. Available at: <https://ieeexplore.ieee.org/document/8675730?denied=> [Accessed 17 April 2021].
5. Afifi,  A.;   Alhumam,  A.;Abdelwahab, A. Convolutional NeuralNetwork for Automatic Identificationof Plant Diseases with Limited Data.Plants2021,10, 28.https://dx.doi.org/10.3390/plants10010028
6. https://www.sciencedirect.com/science/article/abs/pii/S0168169920302258

## **Introduction:**
</br> Face recognition has been a topic of research since 1961 and there were many algorithms developed to improve the recognition rate. In earlier times PCA was used for face classification. There has been a great transformation for feature extraction when N. Dalal and B. Triggs have introduced the concept of Histograms of Oriented Gradient for human detection in [6]. In PCA we use eigen vectors as features while in HoG we will use feature vectors which will be extracted with the help of oriented gradients and all the mathematical analysis has been greatly explained in article 'Improved Face Recognition Rate Using HOG Features and SVM Classifier' [5]. There has been a detailed and intuitive article on HoG feature extraction to get strong intuition on how image matrix will get converted to HoG feature vector in [2]. After extracting features we will use Support Vector Machine with different image size and K-Nearest Neighbor model to predict the class of the face. 
</br>
## **Approach:**
We have followed the basic image classification approach with the help of 2 classifiers and then comapred the results based on the accuracy we get and the time model takes to get learn. The flow of image classification starts with partitioning the dataset (Dataset: Fetch lfw people) into training and testing folders and then carry out the feature vectors for each image in training as well as in testing folder and terminate the process by classifying the image [5].
</br>
We will now discuss the approach taken to extract our feature vectors using HoG. The HoG feature discriptor focuses on the shape and edges of the image, along with that it also focus on the direction and orientation of the edge and because of this it is not similar to other edge detectors. The key idea behind this is to divide the image into small portions and for each portion we will calculate gradient and orientation. After getting small region we will calculate gradient of each pixel by simply subtracting the neighbor pixels in up-down and side-ways manner. After that we need to calculate the magnitude by the gradients in x and y directions using formula <img src="https://latex.codecogs.com/svg.latex?magnitude=\sqrt{(G_x)^2+(G_y)^2}" title="\Large magnitude=\sqrt{(G_x)^2+(G_y)^2}" />
and orientation can be found using <img src="https://latex.codecogs.com/svg.latex?tan(\phi)=\frac{G_y}{G_x}" title="\Large tan(\phi)=\frac{G_y}{G_x}" />
[1],where Gx is the gradient in horizontal direction and Gy is the gradient in the vertical direction [2]. 
</br> After getting the magnitude and orientation will be add to the histogram with scale = 20 and we will get 9 elements from one feature vector of one 8x8 image region which will be return in bin variable if we use HoG inbuilt function [2]. With the help of these feature vectors we will train our classifier and test on the testing data. After testing the model we will compare the values of predicted and ground truth to get the accuracy of our model. 
</br>
## **Results:**


## **References:**
1. https://stackoverflow.com/questions/11256433/how-to-show-math-equations-in-general-githubs-markdownnot-githubs-blog
2. https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
3. https://scikit-learn.org/stable/modules/svm.html
4. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
5. Dadi, H. and Pillutla, G., 2021. Improved Face Recognition Rate Using HOG Features and SVM Classifier. [online] www.iosrjournals.org. Available at: <https://www.researchgate.net/profile/Pg-Mohan/publication/305709603_Improved_Face_Recognition_Rate_Using_HOG_Features_and_SVM_Classifier/links/57aedbba08ae95f9d8f11b57/Improved-Face-Recognition-Rate-Using-HOG-Features-and-SVM-Classifier.pdf> [Accessed 17 April 2021].
6. Dalal, N. and Triggs, B., 2021. Histograms of oriented gradients for human detection. [online] Ieeexplore.ieee.org. Available at: <https://ieeexplore.ieee.org/abstract/document/1467360> [Accessed 17 April 2021].

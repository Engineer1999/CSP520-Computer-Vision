# Introduction
Face recognition has its applications in many fields like multimedia, entertainment, security, etc. It basically focuses on the problem of correctly identifying faces in images and correlate them with the faces in the dataset. Scale Invariant Feature Transform (SIFT) is one of the feature extraction method that is further used for matching different views of an object. The features extracted via SIFT are invariant to image scale and rotation. Here, the SIFT algorithm is used for face recognition.
SIFT is a key point detector as well as feature descriptor. It is a feature detection algorithm, which has its applications in Computer Vision domains like image matching, object detection, etc. It transforms image data into scale invariant coordinates. It extracts unique invariant feature and is robust to affine distortion (rotation, scale, shear), 3D viewpoint change, noise and illumination.
The SIFT process consist of the following four steps:
1) Scale space peak selection (Potential location for features): To make sure that the features are scale independent
2) Key point localization (Locate the key points): To identify suitable features or keypoints
3) Assign Orientation to key points: To make sure that the keypoints are rotation invariant
4) Describe key point as 128 dimensional SIFT descriptor: To assign a unique fingerprint to each keypoint [1]

# Approach 
The [dataset](https://www.kaggle.com/kasikrit/att-database-of-faces) used here, consists of 10 images of 40 subjects each, which is manually splitted into 80% train data and 20% test data, by selecting randomly from the original dataset, while keeping a balance in the number of images of each subject in both train and test data. Keypoints and descriptors are generated for each image using the detectAndCompute function of the initialized SIFT object. A dictionary is created for mapping image ids. In this case, the naming convention of the image files play a major role. The naming convention used here, follows a particular pattern: subjectId_s<ID(1...10)>. Descriptors play the role of features, for the training data. Support Vector Classifier, with RBF (radial basis function) kernel is used with regularization parameter as 10, and kernel coefficient as 0.00001, to fit the model. Keypoints and descriptors for each image in test data are computed and labels are predicted using the SVC classifier. The occurence of each predicted value (labels assigned to keypoints) are counted and the subject id is assigned based on maximum of the keypoints. The accuracy is then computed as the number of true predicted subject ids out of total. 

Equations regarding the same are added in "[images](https://github.com/yeshaajudia/CSP520-Computer-Vision/tree/main/Assignment%202%20%26%203/images)" folder. 

# Results
Accuracy of the model is considered as a performance metric for the experiment conducted. The ROC curve for the same is also plotted, which shows that the working of the
model is good and the accuracy also turns out to be 98%. The area under the ROC curve is near to 1, which supports the previous statement. SVM classifier proves to be effective in high dimensions and works well even when the number of dimensions is greater than the number of samples. It also provides the facility to use different kernel functions for various decision functions.

![ROC.JPG](https://github.com/yeshaajudia/CSP520-Computer-Vision/blob/main/Assignment%202%20%26%203/images/ROC.JPG)

# Installation guidelines and platform details
The experiment includes creating a SIFT object using the SIFT_create function. Since the SIFT function is not supported in the latest version, we need to install the older version of opencv, using the following commands:

!pip install opencv-python==3.4.2.17

!pip install opencv-contrib-python==3.4.2.17

The above installations are provided in the requirement.txt file.
Some other libraries used are cv2, numpy, matplotlib, sklearn, etc. The experiment was conducted on google colab. 

# References 
[1] Lowe, David G. ”[Distinctive image features from scale-invariant keypoints](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.313.1996&rep=rep1&type=pdf).” International journal of computer vision 60.2 (2004): 91-110.

[2] Kortli, Yassin, et al. ”[A comparative study of CFs, LBP, HOG, SIFT, SURF, and BRIEF techniques for face recognition](https://www.researchgate.net/profile/Yassin-Kortli/publication/324850900_A_comparative_study_of_CFs_LBP_HOG_SIFT_SURF_and_BRIEF_techniques_for_face_recognition/links/5e2f23d8299bf10a65978843/A-comparative-study-of-CFs-LBP-HOG-SIFT-SURF-and-BRIEF-techniques-for-face-recognition.pdf).” Pattern Recognition and Tracking XXIX. Vol. 10649. International Society for Optics and Photonics, 2018.

[3] Rani, Ritu, Surender Kumar Grewal, and Kuldeep Panwar. ”[Object Recognition: Performance evaluation using SIFT and SURF](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.403.502&rep=rep1&type=pdf).” International Journal of Computer Applications 75.3 (2013).

[4] Ozturk, Saban, and A. Bayram. ”[Comparison of HOG, MSER, SIFT, FAST, LBP and CANNY features for cell detection in histopathological images](https://www.researchgate.net/profile/Saban-Oeztuerk/publication/324983531_Comparison_of_HOG_MSER_SIFT_FAST_LBP_and_CANNY_features_for_cell_detection_in_histopathological_images/links/5af00461a6fdcc8508b95d77/Comparison-of-HOG-MSER-SIFT-FAST-LBP-and-CANNY-features-for-cell-detection-in-histopathological-images.pdf).” Helix 8.3 (2018): 3321-3325.

[5] Aljutaili, Daliyah S., et al. ”[A Speeded up Robust Scale-Invariant Feature Transform Currency Recognition Algorithm](https://www.researchgate.net/profile/Dina-Hussein-4/publication/328118510_A-Speeded-up-Robust-Scale-Invariant-Feature-Transform-Currency-Recognition-Algorithm/links/5bb8765b92851c7fde2f38c6/A-Speeded-up-Robust-Scale-Invariant-Feature-Transform-Currency-Recognition-Algorithm.pdf).” International Journal of Computer and Information Engineering 12.6 (2018): 365-370

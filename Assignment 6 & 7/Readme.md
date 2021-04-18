# Assignment-6-7

# Introduction
The process of taking content and a style image as input and outputting an image that has the content same as content image along with style of the style image is known as Neural Style Transfer. This is made possible with the help of a popular deep learning algorithm known as convolutional neural network (CNN). This paper will start by applying major techniques for neural style transfer using VGG16 and VGG19 pretrained models.

# Approach
For implementation, we have made use of VGG16 and VGG19 architecture which is a pretrained image classification network. These CNN architectures are improvements over AlexNet with kernel-sized filters (11 filters in layer one and 5 filters in layer two) getting replaced with multiple 3×3 kernel-sized filters one after another [14].
	
![image](https://user-images.githubusercontent.com/82309888/115151053-e00ee100-a088-11eb-8e5b-68d3b5d4b2cc.png)

In this deep neural network, the initial few layers starting from the network’s input layer generate activations that represent the features such as edges, texture and other low level features while the final layers generate activations representing features such as eyes, nose, car wheel and other high level features. The input is 224*224 RGB image to Conv1 layer [14]. This is passed through multiple receptive layers with size 3*3 so as to capture the smallest features. 1*1 convolution filters are used for performing dimensionality reduction. Stride 1 and padding of 1 pixel is used. There is spatial pooling of 5 max pool layers performed on a 2*2 window with stride 2.

![image](https://user-images.githubusercontent.com/82309888/115151057-e8671c00-a088-11eb-9700-13ca2cbc8e66.png)

There are three fully connected layers wherein the first one has 4096 channels and second one has 1000 channels respectively. Third layer is the Softmax layer. Each hidden layer is equipped with a rectification (ReLU) non linearity [14]. It is the intermediate layers that help defining the content and style of an image. While there are 16 deep neural layers in VGG16, there are 19 dense layers in VGG19.

Initially there were two images taken, one being the content image and the other one being the style image. Content image was the one to be styled artistically using the style image. These images were first loaded and their dimension was restricted to 512 pixels. Next, the VGG16 and VGG19 models were respectively loaded and their intermediate layers were selected so as to represent the style and content. While first few layers represented low level features and last few layers represented high level features, intermediate layers were used. These were selected as for image classification at high level, understanding image and drawing an internal representation that would take raw pixels of image and convert them to high level complex features was needed. The VGG16 or VGG19 model is next built which returns a list of intermediate layers.

Now, the content image is represented by values of these intermediate feature maps. The style can be represented by means and correlation across these various feature maps. Gram matrix, which is the outer product of feature vector with itself at every location and an average of this outer product over all of these locations, is calculated as follows:

![image](https://user-images.githubusercontent.com/82309888/115151079-fe74dc80-a088-11eb-83da-cadca312de51.png)

The style and content are then extracted and the style transfer algorithm is run. Mean square error of image’s output relative to the target is compared and a sum of losses is taken. Adam optimizer is used for optimization.


# Dataset Details

The dataset used is named CIFAR-100 [4]. This dataset consists of tiny images of objects such as apple, bed, baby, bird, cat, deer, dog, ship, truck etc. There are 100 classes each containing 600 images present in this dataset. Each image has 3 RGB colour channels and a pixel dimension 32*32 which makes the overall size per input equal to 32*32*32 that is 3072. For each class there are 500 training images and 100 testing images. Hence in all, the dataset has 50K training images and 10K testing images. These images were collected by hey Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. In this dataset, all he 100 classes are divided into 20 groups of superclasses. Hence, each image has a “fine” label associated with the class it belongs to and a “coarse” label associated with the “superclass” it belongs to.

# Results
**Image 1: Yellow Labrador 700 × 577 pixels**

![image](https://user-images.githubusercontent.com/82309888/115151213-8b1f9a80-a089-11eb-8332-20173919f656.png)
 
 Fig. 4. Content Image 1
 
![image](https://user-images.githubusercontent.com/82309888/115151224-95419900-a089-11eb-9edd-18cb438f9ca2.png)
 
 Fig. 5. Style Image 1
     
![image](https://user-images.githubusercontent.com/82309888/115151260-bdc99300-a089-11eb-903d-61740465e558.png)
 
 Fig. 6. Augmented Image VGG16
 
 ![image](https://user-images.githubusercontent.com/82309888/115151272-c6ba6480-a089-11eb-8d8b-11b6495d1c39.png)
 
 Fig. 7. Augmented Image VGG19
 
 **Image 2: Paris 910 × 607 pixels**
 
 ![image](https://user-images.githubusercontent.com/82309888/115151285-d639ad80-a089-11eb-9c3a-417533ef3757.png)
 
 Fig. 8. Content Image 2 
  
 ![image](https://user-images.githubusercontent.com/82309888/115151303-e8b3e700-a089-11eb-8953-28b502695f48.png)
 
 Fig. 9. Style Image 2
  
  ![image](https://user-images.githubusercontent.com/82309888/115151310-f79a9980-a089-11eb-8f04-5c33962f9c56.png)
 
 Fig. 10. Augmented Image VGG16    
  
  ![image](https://user-images.githubusercontent.com/82309888/115151324-03865b80-a08a-11eb-985f-ea8bc7f86dbf.png)
 
 Fig. 11. Augmented Image VGG19
  
  # Installation Guidelines and Platform Details:

  Code has been run over Google Colab. Images have been used directly using Google API link provided in the reference section. 
  
  # References
  
1.	Luan, Fujun, et al. ”Deep photo style transfer.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. 
2.	 Nikulin, Yaroslav, and Roman Novak. ”Exploring the neural algorithm of artistic style.” arXiv preprint arXiv:1602.07188 (2016). 
3.	 Mikołajczyk, Agnieszka, and Michał Grochowski. ”Style transfer-based image synthesis as an efficient regularization technique in deep learning.” 2019 24th International Conference on Methods and Models in Automation and Robotics (MMAR). IEEE, 2019.
4.	Simonyan, K. and Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
5.	Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009).  Imagenet: A large-scale hierarchical image database. In Computer Vision
and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pages 248–255. Ieee.
6.	Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. ”A neural algorithm of artistic style.” arXiv preprint arXiv:1508.06576 (2015).
7.	Fei-Fei, L., Fergus, R., and Perona, P. (2006). One-shot learning of object categories. IEEE transactions on pattern analysis and machine intelligence, 28(4):594– 611
8.	Griffin, G., Holub, A., and Perona, P. (2007). Caltech-256 object category dataset. 
9.	Vishwakarma, Dinesh Kumar. ”A State-of-the-Arts and Prospective in Neural Style Transfer.” 2019 6th International Conference on Signal Processing and Integrated Networks (SPIN). IEEE, 2019.
10.	Wu, Yong, et al. ”Convolution neural network based transfer learning for classification of flowers.” 2018 IEEE 3rd international conference on signal and image processing (ICSIP). IEEE, 2018.
11.	Xu, Yijie, and Arushi Goel. ”Cross-Domain Image Classification through Neural-Style Transfer Data Augmentation.” arXiv preprint arXiv:1910.05611 (2019).
12.	Rohit Thakur, ”Step by step VGG16 implementation in Keras for beginners”, Towards Data Science, August 2019.
13.	Zheng, Yufeng, Clifford Yang, and Alex Merkulov. ”Breast cancer screening using convolutional neural network and follow-up digital mammography.” Computational Imaging III. Vol. 10669. International Society for Optics and Photonics, 2018.
14.	Shaha, Manali, and Meenakshi Pawar. ”Transfer learning for image classification.” 2018 Second International Conference on Electronics, Communication and Aerospace Technology (ICECA). IEEE, 2018.



                                                              
 	 
                                  


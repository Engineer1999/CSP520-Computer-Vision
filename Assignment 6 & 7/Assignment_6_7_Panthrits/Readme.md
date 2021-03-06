# Neural Style Transfer (Assignment 6 & 7)

## Introduction

Converting a normal image into some form of artistic style image has been an area of interest for many decades. Neural style Transfer is deep learning technique to get an output image having the content of ”content image” and style of ”style image”, so that it feels like out content image has been stylized.
Many apps like DeepArt and Prisma use this techinque. Many artists and designers around the globe use this method to generate new artwork based on old developed Styles.

## Approach

-  Loading Content And style Images
- Deciding the hyper parameters of all the losses , i.e total loss weight, content loss weight, style loss weight.
- Resizing image into desired dimensions.
- Pre-processing content,style and combined image
- Defining content and style layers
- Defining gram matrix to be used in style loss
- Defining content loss,style loss and the total loss
- Reducing Content loss MSE between content and combined images and reducing style loss between style and combined images using stoichastic gradient descent.
- De-processing the combined image and Saving.

## Results

We took 4 styles as shown below and produced the output image on the image of a cat. Each style was used 4 times for constructing the Image with one dimension as 400 and one as 512 on both models VGG16 as well as VGG19.
- Original Image:
<img src=cat2.jpg width="400">
- Style 1
Here we have taken the famous painting of the Great Wave of Kanagawa as our styling.
<img src=wave.jpg width="400">
<img src=cat_wave_19_512.png width="400">
- Style 2
Here we have taken the an abstract color art as our styling. And we can see that the reproduced image is a mixture of colors
<img src=style23.jpg width="400">
<img src=cat_style23_19_512.png width="400">
- Style 3
In this we have taken styling as an image of a folded book but the results obtained were interesting.
<img src=fire.jpg width="400">
<img src=cat_fire_19_512.png width="400">
- Style 4
Here we have used a Picasso Classic as our styling but the results obtained were quiet similar to the input image. But we noticed that the reproduced image has a box like structure in it just like the style image.
<img src=Pablo.jpg width="400">
<img src=cat_Pablo_19_512.png width="400">

## Installation
The Coding was done on google colaboratary using free gpu available.
Here are some of the dependencies that u may need to install while using this code in the local enviornment:
- Numpy
- Tensorflow
- Keras

Please refer to requirnments.txt file for more details. 
Go  to the directory where requirements.txt is located. <br />
Activate your conda enviornment. 
Run: conda install -r requirements.txt in your shell.


## References
- [Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [Intuitive Guide to Neural Style Transfer](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee)
- [Art AI: The Logic Behind Deep Learning Style Transfer](https://medium.com/codait/art-ai-the-logic-behind-deep-learning-style-transfer-1f59f51441d1)

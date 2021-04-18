## Introduction 
Imagine if you can see yourself in the style of a lion or your favourite cartoon character or any other theme you choose. It may be a dream at this point. But not if you know a little about deep learning. Neural style transfer is a technique that showcases the capabilities of a deep neural network.It is an art of giving a particular style to any image, which is illustrated further in this document. Wish you enjoy exploring it. 
<br/>
Content and Style images:
<br/>
<img src=r1.png width="400">
<br/>
Target image:
<br/>
<img src=rr1.png width="400">

## Approach
Initially we took two images as inputs: style and contentimage.
- We  created  a  model  using  the  pre-trained  weights  ofVGG16.
- Weusedthelayers:block1conv1,block2conv1,block3conv1,block4conv1,block5conv1asstylelayers.
- We used the layers:block4conv3 as content layer.
- We pre-processed the content and style images using theircorresponding layers.
- We  computed  loss  between  the  output  of  content  layerand white noise image. This loss is the content loss.
- For  the  style  loss,  we  computed  the  feature  correlationsbetween the feature maps(output of style layers).
- TotalLoss=α×ContentLoss+β×StyleLoss.
- Then  we  performed  gradient  descent  on  a  white  noiseimage  to  find  another  image  that  matches  the  featureresponses of the original image


## Installation Guidelines and Platform Details
Run in Google Colab
There is a CV_Ass6-7.ipynb file in the code section. You can directly open this jupyter notebook file in google colab and run the code by taking two images in input as the content image and the style image.

## Run in Jupyter Notebook
- First create an environment using anaconda
- Then install tensorflow using the command pip install tensorflow.
- Install pytorch, torchvision, PIL using pip. 
- For better performance you can install cuda toolkit and run the sectio section of the ipynb file.
- Open the given ipynb file and change the content image and the style image locations.

## Results
<img src=rr1.png width="400">
<img src=rr2.png width="400">
<img src=rr3.png width="400">

## References


- [Neural Artistic Style Transfer: A Comprehensive Look](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199)
- [Neural Style Transfer(paper)](https://arxiv.org/abs/1508.06576)


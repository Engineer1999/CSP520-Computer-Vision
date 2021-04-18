# Neural Style Transfer

## Introduction

Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

The principle of neural style transfer is to define two distance functions, one that describes how different the content of two images are, Lcontent, and one that describes the difference between the two images in terms of their style, Lstyle. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image.

## Approach

Our approach broadly included of the following steps: 

1. Data Visualization
2. Basic Data Pre-processing
3. Set up loss functions
4. Create Model
5. Optimization


### Content and Style Representation

In order to get both the content and style representations, we looked at some intermediate layers within our model. Intermediate layers represent feature maps that become increasingly higher ordered as we go deeper. We are used 
the network architecture of VGG16 and VGG19 pretrained image classification network. These intermediate layers are necessary to define the representation of content and style of our images. For an input image, we tried to match the corresponding style and content target representations at these intermediate layers.

### Model

We used both, VGG16 and VGG19 as our image classification CNN. This allowed us to extract the feature maps of the content, style and generated images. In order to access intermediate layers corresponding to our style and content feature maps, we get the corresponding outputs by using Keras Functional API to define our model.
Simply put, we created a model that will take an input image and output the content and style intermediate layers.

### Loss Functions

We have defined 2 loss functions: **content** loss function and **style** loss function. The content loss function ensures that the activations of the high layers are similar between the content image and the generated image. The style loss function makes sure that the
correlation of activations in all the layers are similar between the style image and the generated image.

#### Content Loss Function

The content loss function makes sure that the content present in the content image is captured in the generated image. Since, CNNs capture information about the content in the higher layers, we used the top-most CNN layer to define the content loss function.
[!alt text]()

#### Style Loss Function

Style loss function is trickier than the content loss function. To extract the style information from the VGG network, we use all the layers of the CNN. Furthermore, style information is measured as the amount of correlation present between features maps in a given layer. Next, a loss is defined as the difference of correlation present between the feature maps computed by the generated image and the style image. Mathematically, the style loss is defined as,
IMAGE

#### Final Loss

The final loss is defined as,
IMAGE

where alpha and beta are hyperparameters.

### Optimizer

We have used the Adam Optimizer in order to minimize our loss. We iteratively update our output image such that it minimizes our loss: we don’t update the weights associated with our network, but instead we train our input image to minimize loss. 

## Results

## Platform 

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)

## Installation

- Clone this repository using 
```
git clone https://github.com/Engineer1999/CSP520-Computer-Vision.git
cd CSP520-Computer-Vision
```
- Install the dependencies using
```
pip install -r requirements.txt
```
- To run locally, launch jupyter notebook using `$ jupyter notebook` or upload the `.ipynb` file on Google Colab.


## References

1. https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199
2. https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199
3. https://www.google.com/url?sa=D&q=https://medium.com/codait/art-ai-the-logic-behind-deep-learning-style-transfer-1f59f51441d1&ust=1618830720000000&usg=AOvVaw1PLIoi6v-x-30I_X4tpN4V&hl=en

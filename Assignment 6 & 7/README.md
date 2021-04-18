Introduction
In this assignment, we have performed neural style transfer which takes in a content and a style image and gives an image which is a combination i.e., a content image painted in the form of the style image, as output. Here, the intermediate layers of pretrained VGG16 and VGG19 models are used for style transfer. The main objective of the assignment was to understand the impact of parameters on the style transfer result. TensorFlow’s eager execution is used that evaluates operations immediately, without building graphs. The results are generated for different style images.

Approach (Visual explanation along with description is preferred)
The VGG16 and VGG19 pretrained models are loaded and their intermediate layers are used to generate the output from them. First convolution layer of block5 is taken as the content layer, and the first convolution layers of the five blocks are taken as the style layers. Here, the layers are not needed to be trained. The content and the style images are preprocessed and the content and style feature representations are extracted. Adam optimizer is used with a learning rate of 5, content weight is α = 103, and
style weight β = 10−2. The image is then mean normalised. The content loss is the Mean Squared Loss and the style loss is computed using Gram Matrix. Style score and content score are computed based on the style loss and content loss respectively. Both the scores are multiplied with their respective weights and total loss is computed by adding them. Gradients are computed with respect to the input image. The image with the minimum total loss, obtained while iterating through, is considered as the final output of the style transfer.

Results
The content loss and style loss are minimized. Decreasing the dimension of the image, makes the style image, content image and the final output image blur. Also, it is observed that a higher dimension image takes much time to generate the result, whereas an image with lower dimension generates the result within less time. We have used different style images to generate the result, for both, VGG16 nad VGG19.
<images>

Installation guidelines and platform details
This experiment was carried out on google colab, using GPU runtime. Python libraries such as matplotlib, numpy, tensorflow, etc are used. 

References with necessary hyperlinks.
[1] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. ”A neural algorithm of artistic style.” arXiv preprint arXiv:1508.06576 (2015).
[2] Huang, Xun, and Serge Belongie. ”Arbitrary style transfer in realtime with adaptive instance normalization.” Proceedings of the IEEE International Conference on Computer Vision. 2017.
[3] Yuan, Raymond. Neural Style Transfer: Creating Art with Deep Learning Using Tf.keras and Eager Execution, 3 Aug. 2018, medium.com/tensorflow/neural-style-transfer-creating-art-with-deeplearning-using-tf-keras-and-eager-execution-7d541ac31398.
[4] Desai, Shubhang. Neural Artistic Style Transfer: A Comprehensive Look, 14 Sept. 2017, medium.com/artists-and-machineintelligence/neural-artistic-style-transfer-a-comprehensive-lookf54d8649c199.
[5] Kasten, N. (2019, February 22). Art and AI: The Logic Behind Deep Learning ‘Style Transfer’ [Web log post]. Retrieved from https://medium.com/codait/art-ai-the-logic-behind-deep-learning-styletransfer-1f59f51441d1
[6] Ganegedara, T. (2019, January 9). Intuitive Guide to Neural Style Transfer [Web log post]. Retrieved from https://towardsdatascience.com/lighton-math-machine-learning-intuitive-guide-to-neural-style-transferef88e46697ee
[7] Ghiasi, Golnaz, et al. ”Exploring the structure of a real-time, arbitrary neural artistic stylization network.” arXiv preprint arXiv:1705.06830 (2017).
[8] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. ”Image style transfer using convolutional neural networks.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[9] Simonyan, Karen, and Andrew Zisserman. ”Very deep convolutional networks for large-scale image recognition.” arXiv preprint arXiv:1409.1556 (2014).
[10] Nepal, Prabin. VGGNet Architecture Explained, 30 July 2020, medium.com/analytics-vidhya/vggnet-architecture-explainede5c7318aa5b6.
[11] Patel, Khush. Architecture Comparison of AlexNet, VGGNet, ResNet, Inception, DenseNet, 8 Mar. 2020, towardsdatascience.com/architecturecomparison-of-alexnet-vggnet-resnet-inception-densenetbeb8b116866d.
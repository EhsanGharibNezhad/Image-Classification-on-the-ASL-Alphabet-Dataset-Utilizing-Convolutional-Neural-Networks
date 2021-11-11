# Image-Classification-on-the-ASL-Alphabet-Dataset-Utilizing-Convolutional-Neural-Networks

###  Investigator: Ehsan Gharib-Nezhad

<p>
  <a href="https://www.linkedin.com/in/ehsan-gharib-nezhad/" rel="nofollow noreferrer">
    <img src="https://i.stack.imgur.com/gVE0j.png" alt="linkedin"> LinkedIn
  </a> &nbsp; 
  <a href="https://github.com/EhsanGharibNezhad/" rel="nofollow noreferrer">
    <img src="https://i.stack.imgur.com/tskMh.png" alt="github"> Github
  </a>
</p>

<br></br>
# <a id = 'ProblemStatement'>Problem Statement</b></a>

American Sign Language (ASL) is the primary mean of communication of people with hearing disabilities and it plays a crucial role in the Deaf community in the north America. Hence, recognition of ASL with computer algorithims is a continoius effort and challenge in the field of machine learning due to its complexity. In this project, more than 11,000 images of all ASL alphabets are preccessed, vectorized, and resized using Pillow, Numpy, and sklearn libraries. Then, Convolutional Neural Networks (CNNs) is build and trained using all these processed images. 
Different features of CNNs architectures such as filters, hidden layers, and number of nurons are assessed in order to leverage the accuracy and F1-score metrics. The trained CNNs model recongize the train and test subsets with >98% of accuracy. 

---

<br></br>
# <a id = 'Content'> Content </b></a>

- [Problem Statement](#ProblemStatement)
- [Content](#Content)    
- [Repo Structure](#RepoStructure)    

    - [Data Dictionary](#ddict)
    - [Background](#Background)
    - [1. Image Processing](#ImageProcessing)
   	- [2. Methodology](#Methodology)
    	- [2.1 Overview](#Overview)
    	- [2.2 Feedforward Neural Networks (FNNs)](#FNNs)
    	- [2.3 Convolutional Neural Networks (CNNs/ConvNets)](#CNNs)
    - [Results](#Results)    
    - [Conclusion](#Conclusion)
    - [Recommendations](#Recommendations)
    - [References](#references)



---
# <a id = 'RepoStructure'> Repo Structure </b></a>
## notebooks/ <br />

*Image Processing and models:*\
&nbsp; &nbsp; &nbsp; __ [image_processing_NeuralNetwork_modeling.ipynb](notebooks/image_processing_NeuralNetwork_modeling.ipynb)<br />

## datasets/<br />
*Processed/Vectorized/Resized Images from sub Reddits:*\
&nbsp; &nbsp; &nbsp; __ [preprocessed_covid19positive_reddit_LAST.csv](datasets/preprocessed_covid19positive_reddit_LAST.csv)<br />

## outputs/<br />
*Stored CNNs model:*\
&nbsp; &nbsp; &nbsp; __ [cnns_model.h5](outputs/cnns_model.h5)<br />
&nbsp; &nbsp; &nbsp; __ [cnns_model.json](outputs/cnns_model.json)<br />


[README.md](README.md)<br />


---

# <a id = 'ASLdataset'>American Sign Language Image Dataset</b></a>
The image data set is taken from [Kaggle database](https://www.kaggle.com/grassknoted/asl-alphabet) and consists of 87,000 images in 29 folders which represent 26 American sign language classes (e.g., A, B, ..., Z) and 3 classes for SPACE, DELETE and NOTHING.

# <a id = 'ImageProcessing'>Image Processing</b></a>
Image preparation and processing is an important part when training a convolutional neural network. The following tasks are carried out to prepare the raw RGB sign language image: 

1. Loading *.jpg* images using [Python Pillow library](https://pypi.org/project/Pillow/) (PIL)

2. Converting images to Numpy array: 
We need to first convert RGB images into 3d arrays with [height, width, channel] dimentions. Numpy vectorized-images are the main format we impliment operations in machine learning and neural networks. Loaded images by Pillow library are converted to matrices in this step using *numpy.asarray* tool.   


3. Resizing/downscaling the images: The vectorized image arrays have 200 by 200 dimentions and 3 RGB channels (Red, Green, Blue). Since the background image is roughly distinctive from the sign alphabet, these dimentional looks overwholming and increase the computational time dramatically. In addition, 1 channel would be sufficient for neural networks to identify the alphabet and classify them. Hence, [scikit-image](https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html) library tool (*skimage.transform.resize*) is utilized to reduce the image arrays. 

All these image processing tools are embedded into the following function in the main code:

<img width="1118" alt="image" src="https://user-images.githubusercontent.com/22139918/141200677-f5744261-b05a-4846-aacc-23a4bd45b080.png">

![image](https://user-images.githubusercontent.com/22139918/141347858-ad87fc39-6483-4f30-8b46-e97f979ab767.png)

--- 

# <a id = 'Methodology'>Methodology</b></a>

## <a id = 'Overview'>Overview</b></a>

**Archetacture Overview:** Convolutional Neural Networks consists of a number of neurons that are fed with input vectorized and resized image data, and includes weights and biases.  Neurons are trainable and learnable because each of them performs a non-linear operation on the input arrays through ReLU function [[Ref]](https://cs231n.github.io/convolutional-networks/#overview). ReLU (Rectified Linear Unit) is ideal for this deep learning purpose because Rectified Linear Unit takes less comutational time to be trained and hence this will reduce the optimization time of the neural network in the gradient descent surface [[Ref]](https://www.mygreatlearning.com/blog/relu-activation-function/). The overal score of the neural network then can be expressed with a single metric: Loss fuction [[Ref]](https://towardsdatascience.com/understanding-different-loss-functions-for-neural-networks-dd1ed0274718).


## <a id = 'FNNs'>Feedforward Neural Networks (FNNs)</b></a>

**Feedforward Neural Networks (FNNs):** In a regular neural network, a certain amount of input data is fed to the neurons in the input layer and then the information iterate through the neural network in forward and backward directions in an attempt to better train the neurons to reduce the loss function. However, this method is not specilized to learn from images and could takes a long time to be completed. In case of image dataset, each input in this project has 60x60x1=3,600 (with,length, 1 color channel) length. This array of data from the whole set of the training images looks overkilling and requires a very long computational time to be managed when it goes into a single neuron. Given the multiple hidden layers and large number of neurons in each layer, regual neural network would not be a great choice to feed the data with and gain a high accuracy score.    

![image](https://user-images.githubusercontent.com/22139918/141359354-83aece64-5cc6-449d-9985-ea4ecbc29f72.png)

![image](https://user-images.githubusercontent.com/22139918/141365776-473bee30-5565-4eef-9532-d7971682a5df.png)


## <a id = 'CNNs'>Convolutional Neural Networks (CNNs/ConvNets)</b></a>

**Convolutional Neural Networks (CNNs/ConvNets):** Compared to the the regular FNNs archetature, CNNs has extra layers to apply further processing to the input images through Convolution operator, Maxpooling, and Padding. The central idea of the convolution layers is to extract the importnat features from the image and simplify them (or downscale) them. Convolution layer consists of a set of filters that take the original image and convolve them the the kernel. The following figure shows the impact of the applying 3x3 kernel filter (left), 2x2 Max Pooling (middle), and 3x3 Max Pooling (right) on a 2D vectorized image ([[Fig credit]](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)). 

<img width="1145" alt="image" src="https://user-images.githubusercontent.com/22139918/141344082-d075edfe-4160-4806-9c66-fc2d298e00d1.png">

In the following, the general script for the build CNNs is represented which includes 2 hidden layers (with 384 and 128 neurons). In addition, one 2D convolutional layer with 6 filters (3x3 kernel) is included. MaxPooling2D and Padding operations are tested but did not show significat impact. 


![image](https://user-images.githubusercontent.com/22139918/141359111-1d4a72d6-733f-4e66-9c55-5b2299a1dadd.png)

![image](https://user-images.githubusercontent.com/22139918/141367817-65a08986-68e4-42a8-9d0a-3b53e6b6df6c.png)


---
# <a id = 'Results'>Classification Results from CNNs model</b></a>

![image](https://user-images.githubusercontent.com/22139918/141369073-09ccdcd2-f85d-4025-8463-5fbe6e92bc84.png)


![image](https://user-images.githubusercontent.com/22139918/141369000-30658295-7803-4372-a493-f88429d7b0b0.png)



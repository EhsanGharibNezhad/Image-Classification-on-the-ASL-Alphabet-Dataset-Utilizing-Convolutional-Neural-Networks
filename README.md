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
   	- [2. Convolutional Neural Networks (CNNs)](#CNNs)
    	- [X](#X)
    	- [Y](#Y)
    - [Methodology](#Methodology)    
    	- [Sign Language Classifier](#SignLanguageClassifier)	
    - [Exploratory Data Analysis](#eda)    
    - [Results](#Results)    
    - [Conclusion](#Conclusion)
    - [Recommendations](#Recommendations)
    - [References](#references)



---
# <a id = 'RepoStructure'> Repo Structure </b></a>
## notebooks/ <br />

*Setp 1: Image Processing:*\
&nbsp; &nbsp; &nbsp; __ [1__data-collectionPrepPandemic.ipynb](notebooks/1__data-collectionPrepPandemic.ipynb)<br />

*Setp 2: Exploratory Data Analysis:*\
&nbsp; &nbsp; &nbsp; __ [3__ExploratoryDataAnalysis_EDA.ipynb](notebooks/3__ExploratoryDataAnalysis_EDA.ipynb)<br />

*Setp 3: CNNs Models: Classifiers*\
&nbsp; &nbsp; &nbsp; __ [4-7__model_Adaboost.ipynb](notebooks/4-7__model_Adaboost.ipynb)<br />

## datasets/<br />
*Processed/Vectorized/Resized Images from sub Reddits:*\
&nbsp; &nbsp; &nbsp; __ [preprocessed_covid19positive_reddit_LAST.csv](datasets/preprocessed_covid19positive_reddit_LAST.csv)<br />

## output/<br />
*Stored CNNs model:*\
&nbsp; &nbsp; &nbsp; __ [preprocessed_covid19positive_reddit_LAST.csv](datasets/preprocessed_covid19positive_reddit_LAST.csv)<br />



[presentation.pdf](presentation.pdf)<br />

[ReadMe.md](ReadMe.md)<br />

---
---
# <a id = 'ddict'>Data <b>Dictionary</b></a>


|feature name|data type|Description|
|---|---|---|
| selftext |*object*|Original Reddit posts with no text processing|
| subreddit|*object*|Subreddit category: r\Covid19Positive and r\PandemicPreps|
| created_utc|*int64*|Reddit posting date|
| author|*object*|Author ID|
| num_comments|*int64*|Number of comments/reply to that post|
| post|*object*| Reddit post after text precessing with normal/unstemmed words|
| token|*object*| Reddit post after text precessing with word stemming|

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


--- 

# <a id = 'CNNs'>Convolutional Neural Networks (CNNs)</b></a>

### 1. <a id = 'X'>X</b></a>

### 2. <a id = 'Y'>Y</b></a>


# <a id = 'SignLanguageClassifier'>Sign Language Classifier</b></a>


# <a id = 'ExploratoryDataAnalysis'>Exploratory Data Analysis</b></a>


# <a id = 'Results'>Results</b></a>


# <a id = 'Conclusion'>Conclusion</b></a>

# Article about the lecture: 

## Title: Edge Detection and Segmentation in Computer Vision
- This is a medium article that explains the concepts of edge detection and segmentation in computer vision. Explaining the techniques and methods used in edge detection and segmentation.

## Abstract ( Will be seen in the search results)
Why is edge detection and segmentation important in computer vision? What are the different methods and techniques used in edge detection and segmentation? This article explains the concepts of edge detection and segmentation in computer vision.

## Introduction
- When we talk about computer vision, the main 3 tasks are: 
  - Image Classification
  - Object Detection
  - Segmentation

This article will focus on the last task, Segmentation. Segmentation is the process of dividing an image into different segments. When each pixel is classified as a class, it is called segmentation. There are different methods and techniques used in segmentation.

## Is Segmentation a classification problem, clustering problem or a regression problem?


In order to correctly answer this adequately, we need to understand the main premise of the question.

## Classification
- Classification is the process of categorizing the data into different classes where each pixel is classified as a class.

## Clustering
- Clustering is the process of grouping the data into different clusters based on the similarity of the data however, it is not the same as classification as the data is not labeled and the algorithm will decide the classes.

## Regression
- Regression is the process of predicting the continuous value of the data. One might think that there are no continuous values in pixel-wise classification but the moment we convert each pixel to a continous value representing the color of the pixel the problem becomes continuous.

Based on the above explanation, we can conclude that Segmentation has a classification problem as each pixel is classified as a class but also has a regression problem as each pixel is represented as a continuous value and by the end of the process we are clustering the pixels into a class based on the similarity of the pixels in terms of the position 


## Evaluation Metric for Segmentation
Disclaimer: There is not a single magical evaluation metric which can be used for all the segmentation problems. The choice of the evaluation metric depends on the problem and the data. However, there are some common evaluation metrics used in segmentation. 

## DICE Coefficient
The most used evaluation metric in segmentation is the DICE coefficient. It is used to quantify the similarity between two sets.
The Dice coefficient formula calculates the similarity between two sets by measuring twice the intersection of the sets divided by the sum of their sizes. The formula is given by:

$$ \text{Dice}(A,B) = \frac{2|A\cap B|}{|A| + |B} $$

where:
- A is the predicted set
- B is the true set
- |A ∩ B| is the intersection of the predicted and true set
- |A| is the area of the predicted set
- |B| is the area of the true set


## Jaccard Index
The Jaccard Index is another great evaluation metric used to determine the similarity between two sets. Jaccard Index also carries the name of Intersection over Union (IOU). The alternative name is very descriptive of the formula which is given by:
$$ \text{Jaccard}(A,B) = \frac{|A\cap B|}{|\max(A,B)|} = \frac{|A\cap B|}{|A|+|B|-|A\cap B|} $$
where:
- A is the predicted set
- B is the true set
- |A ∩ B| is the intersection of the predicted and true set
- |A| is the area of the predicted set
- |B| is the area of the true set


## Pixel-wise Segmentation
The ground basis of the pixel-wise segmentation lies in the fact that each pixel is classified as a class. The number of classes depends on the problem and how granular the segmentation is. The larger the number of classes, the more granular the segmentation is. The smaller the number of classes, the less granular the segmentation is. By increasing the number of classes the model will be able to detect more details in the image. However, the more classes the more complex the model will be and the more data will be needed to train the model. The number of classes can be warried from 0 to N where: {N ∈ N}

The most simple example of pixel-wise segmentation is a binary mask where the number of classes is 2. The classes are usually the object and the background. In medical imaging, the classes can be the tumor and the background or a specific organ like lungs and the background. By utilizing the pixel-wise segmentation, the model can detect the exact location of the object in the image.


## Random Taxonomy
The concept of Random Taxonomy takes into account the global knowledge of the image. Based on common characteristics of the image, the model can categorize the data. Its usage is a staple in the field of computer vision, biology and many other fields. The Random Taxonomy is used in the following methods:
- Histogram-based thresholding
- Edge-based segmentation: Filters
- Region-based segmentation: KNN, GMM
- Combination of the above methods: Edge-based + Region-based (Canney Edge Detection)


## Segmentation via Classification 
Segmentation via classification utilizes models like
KNN: K-nearest neighbours
SVM: Support Vector Machine
NN: Neural Networks
The features we use in the classification model are:

Voxel values
Voxel Position
Gradient magnitude
Neighboring voxel values

The bery important part of classification are labels. Based on labels we can divide it into 2 categories: Binary (2 classes) and Multi-class (more than 2 classes). The example of Binary classification is the segmentation of the object and the background. The example of Multi-class classification is the segmentation of the object, background, sky, tree, car, road, building, etc. The major issue with segmenetation via classification is the fact that the data requires to be labeled. The labeling process is time-consuming and requires a lot of resources.

## Segmentation via Clustering
Segmentation via clustering utilizes models like:
K-means
GMM: Gaussian Mixture Model

The main process of the algorithm workflow is to pick a K number of clusters and then assign each data point to the nearest cluster. Calculate the respective centroids and then repeat the process until the centroids do not change.

## Thresholding
Thresholding answers the question: Is this a background or a foreground? The thresholding method is the simplest method of segmentation.
The concept of thresholding is to convert the grayscale image to a binary image.


The folowing image shows a histogram of an image. The X-axis represents the pixel values and the Y-axis represents the frequency of the pixel values. On this histogram, we can see the peaks of the pixel values. The valley between the peaks is the threshold value. This distinction in the classes allows us to have clearly separated pixels in the picture and thus having a clear segmentation.

## How to choose the threshold value?
The most well-known method is the Otsu's method which searches the point `t` that minimizes the variance of foreground and background pixel values, weighted by class probabilities. The class probabilities is a count of how many pixels with those colors belongto the foreground, and how many to the background
The formula for the Otsu's method is given by:
$$ \sigma^2_w(t) = w_0(t)\sigma^2_0(t) + w_1(t)\sigma^2_1(t) $$
where:
- t is the threshold value
- w_0(t) is the probability of the background
- w_1(t) is the probability of the foreground
- \sigma^2_0(t) is the variance of the background
- \sigma^2_1(t) is the variance of the foreground

Otsu's method is a great method for thresholding and works especially well for bimodal histograms.

## Connectivity
The concept of connectivity explains to us how the pixels are connected to each other. 
2D connectivity is defined by the 4-connectivity and 8-connectivity. Meaning the central pixel is connected to the 4 or 8 neighboring pixels.
The 4-connectivity is defined by the following pixels:
- (x-1, y)
- (x+1, y)
- (x, y-1)
- (x, y+1)

The 8-connectivity is defined by the following pixels:
- (x-1, y-1)
- (x-1, y)
- (x-1, y+1)
- (x, y-1)
- (x, y+1)
- (x+1, y-1)
- (x+1, y)
- (x+1, y+1)

The 3D connectivity scales the 2D connectivity to the 3D space by adding the z-axis to the equation and therefore allowing to have up to 26 neighboring pixels.
The options are 
- 6-connectivity
- 18-connectivity
- 26-connectivity



## Region Growing
Region Growing is a method of segmentation that is based on the concept of the similarity of the neighboring pixels and trying to find continuous regions.
The Algorithm goes as follows:



## Local Adaptive Thresholding

The basic concept of the Local Adaptive Thresholding is calculating the thresholds separately on different regions of the image. By doing so, we can have a more granular segmentation of the image allowing us to spot potential patterns which on a global level would not be visible.

## Conclusion
As we can see there are many different segmentation methods and techniques. The main premise of a successful workflow is understanding your domain and mastering the knowledge of the data. The choice of the segmentation method depends on the problem and the data. There is no one-size-fits-all solution.
By having a clear understanding of the data and the problem, we can choose the best segmentation method and achieve the best results. The future of segmentation is bright and we can expect many new methods and techniques to come up soon. This is a very exciting time for the field of computer vision and image processing.
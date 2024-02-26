# Types of classification
- Binary classification
- Multi-class classification

## Multi-class classification vs Multi-label classification
- Multi-class classification: Each sample belongs to one and only one class
- Multi-label classification: Each sample can belong to multiple classes

## Segmentation
- Each pixel is classified as a class

## Is Segmentation a classification problem or a regression problem?
- It is a classification problem
- It is a regression problem
- It is clustering problem

## Steps in ML Modelling
- Define the problem
- Collect the data
- Look at the data and explore it
- Decide on the evaluation metric 
 - Classification: Accuracy, Precision, Recall, F1-score, AUC-ROC
 - Regression: Mean Squared Error, Mean Absolute Error, R2 score
 - Clustering: Silhouette score
 - Segmentation
   - Confusion matrix



- Define Features
 - Feature Engineering 
- Split the data into training and testing: Cross-validation
- Train the model
- Evaluate the model
- Tune the model based on Hyperparameters 
- Make predictions  

## Loss functions
- Impurity/Gini Index

## MCC (Matthews Correlation Coefficient)
- IS great for imbalanced datasets
- Takes into account True Positives, True Negatives, False Positives, False Negatives
- Ranges from -1 to 1
- Example 
 
 - Table of Predictions
   - True Positives: 10
   - True Negatives: 20
   - False Positives: 5
   - False Negatives: 5

  - MCC = (10*20 - 5*5) / sqrt((10+5)(10+5)(20+5)(20+5)) = 0.6

## DICE Coefficient
- Mostly used in segmentation
- Used in segmentation
- 2 * (Intersection of Predicted and True) / (Area of Predicted + Area of True)
- ![alt text](imgs/image.png)
- Problem: 


## Jaccard Index
- Used in segmentation
- Intersection of Predicted and True / Union of Predicted and True
- IOU = Overlap / Union


## Segmentation - Pixel-wise categorial labels
- Each pixel is classified as a class
- Example: 
  - Image of a person
  - Each pixel is classified as a class
  - Person, Background, Sky, Tree, Car, Road, Building, etc

- ![alt text](imgs/image-1.png)

## Segmentation Methods
- Random Taxonomy
  - Based on the global knowledge:
     - Histogram-based thresholding
     ![alt text](image-2.png)

  - Edge-based segmentation: Filters

  - Region-based segmentation: 
      - KNN: K-nearest neighbours
      - GMM: Gaussian Mixture Model
          - GMM is a type of model which is used to represent the probability distribution of a random variable
      
  - Combination of the above methods
      - Edge-based + Region-based


## Gaussian kernel
- Used in GMM

## Edge Detection
  -  Edges: Boundary (steep-changes) between two regions 
with distinct gray-level properties.
 - WIll use differentiation to detect edges
- Steepness is defined by spatial derivative:
  - ![alt text](imgs/image-3.png)
- Derivatives with the convolution operation
  - For 2D function, 𝑓(𝑥, 𝑦), the partial derivative is:
    Equation: 
  ![alt text](imgs/image-4.png)

  - For discrete data, we can approximate using finite differences:
    Equation:
    ![alt text](imgs/image-5.png)

- Concept: Calculate the gradient to each pixel in order to detect the edges


## Derivatives Approximation with Kernel Operations
- Robert’s Cross Operator
- Pre-witt Operator
- Sobel Operator
![alt text](imgs/image-6.png)

## Gradient Operators
- 2D first order derivative operators
![alt text](imgs/image-7.png)
- Gradient magnitude: delta f: sqrt((delta f_x)^2 + (delta f_y)^2)
- Gradient direction: tan-1(delta f_y / delta f_x)

## Edge Strength
- Given by the gradient magnitude of the image
- Formula: sqrt((delta f_x)^2 + (delta f_y)^2)

- ![alt text](imgs/image-8.png)

## How do the convolution operations work?
- ![alt text](imgs/image-9.png)

## Edge Detection - Laplacian of Gaussian (LoG)
- LoG is a second derivative operator
- Sum of the second derivatives of the image
- Does not provide the direction of the edge but only the strength
- Very sensitive to noise
  - Apply Gaussian filter to the image before applying LoG
  - Or a Laplacian of Gaussian filter combined: As Gaussian filter is a low-pass filter, it will remove the high-frequency noise and then apply the Laplacian filter to detect the edges in the image

## Derivative of Gaussian
  - ![alt text](imgs/image-10.png)

## Canny Edge Detection: Best Edge Detection Algorithm
- Multi-stage algorithm
- Noise reduction with Gaussian filter
- Edge detection with Sobel operator
- Convolving the image with the derivative of Gaussian
- Find the gradient magnitude 
- Find Gradient orientation
-  Compute Laplacian to find the edges
 
### Hysteresis Thresholding
- Global thresholding: 
  - Thresholding the gradient magnitude
- Regional thresholding:
  - Hysteresis thresholding
    - Two thresholds: High and Low
    - If the gradient magnitude is greater than the high threshold, it is considered as an edge
    - If the gradient magnitude is less than the low threshold, it is not considered as an edge
    - If the gradient magnitude is between the high and low threshold, it is considered as an edge if it is connected to a pixel that is greater than the high threshold

- Parameters to Canney Edge Detection
  - Sigma: Standard deviation of the Gaussian filter
  - High threshold
  - Low threshold
  - Kernel size: Size of the Sobel operator
  - Operator: eg. Sobel, Prewitt, etc
  

# Takeaways
### There are many different segmentation methods
- Thresholding
- Region Growing
- ML-Based
- And even more is coming up soon:
  - Level-sets, 
  - Fast-Marching, 
  - Graph-based
### Edge Detection
- Can produce disconnected boundaries
- Over/Under Segmentation
- Harder in 3D
- Canny smooths before applying gradients
### No one algorithm is the “best”.
- Must match to the problem and data


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
__________


IMAGE

_________

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



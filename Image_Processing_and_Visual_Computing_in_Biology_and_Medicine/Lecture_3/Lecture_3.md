# Lecture 3: Neural Networks - Depp Learning

## PCA
- Reduces the dimensionality of the data

## UMAP
- It is a dimensionality reduction technique

##  Neural Networks

Inputs: A set of features
Outputs: A set of predictions
Weights: A set of parameters that are learned from the data
Loss function: A function that measures how well the model is doing
Optimization: A process of finding the best set of weights


## Concept: 
![alt text](imgs/image.png)
Non-linear function is used to transform the input to the output
Non linear because we want to capture the non-linear relationships in the data
    - ReLU: Rectified Linear Unit
    - Sigmoid
    - Tanh
    - Softmax

## Architecture
![alt text](imgs/image-1.png)
- Input Layer
- Hidden Layer
- Output Layer

- Verifying the quality and adjusting the weights: Backpropagation
- Learning rate: How much to adjust the weights
- Epochs: How many times to go through the data
- Input size: 
    - for classification: Number of features
    - for segmentation: Number of pixels

## Error calculation
- Calclates the distance beween the network's output and the true output
- Each typ eof problem has a different error calculation
    - Mean Squared Error (MSE) – regression
    - Binary Cross Entropy – binary classification
    - Categorical Cross Entropy Loss – multiclass (one-hot-vectors)
    - Negative Log-Likelihood loss – multiclass
    - Multilabel margin loss – multilabel
    - Dice Coefficient – segmentation
    - Jaccard Index – segmentation  

## CNN (Convolutional Neural Networks)
- Used for image data
- Convolutional layers are used to extract features from the image
- The layers are not fully connected
- Filters are used to extract features from the image (edges, textures, etc.)
- Each 2D slice of the filters are called kernel
- Convolutional layers are followed by pooling layers
- Convolutions is a process of applying a filter to an image

### Key characteristics of CNN
- They learn hierarchical representations of the data
- Preserve a spatial relationship between pixels


### Size of input X size of filter
- The size of the input * the size of the filter will give the size of the output
- Input size: 5x5 Filter size: 3x3 Output size: 3x3


### Padding 
- Concpet which is used to preserve the spatial dimensions of the input volume
- It is used to add zeros to the input volume so that the output volume has the same dimensions as the input volume

### Stride
- The number of pixels by which the filter is moved
![alt text](imgs/image-2.png)
n(out) = (n(in) - 2p - k)/s + 1

- n(out) = output size
- n(in) = input size
- p = padding
- k = filter size
- s = stride

### Feature Maps
- The output of the convolutional layer is called a feature map
- The number of feature maps is equal to the number of filters used in the convolutional layer


### Filter Convolutions
- Filters convolve with the 3D input volume to produce 2D output feature maps
- When Using multiple filters, the output will be a 3D output with one 2D feature map per filter

### Activation Function
- ReLU
- Sigmoid
- Tanh
- Softmax

- ReLU is the most commonly used activation function.
- Sigmoid is not good as it can cause the vanishing gradient problem


## Recap
• A convolutional layer convolves each of 
its filters with the input.
• Input -
• a 3D tensor: Width × Height ×
Channels (or Feature Maps)
• Output -
• a 3D tensor: Width × Height × Feature 
Maps (one for each filter)
• Applies non-linear activation function 
(usually ReLU) over each value of the 
output.

Multiple (hyper)parameters to be 
defined: 
• # of filters
• Filters size
• Stride
• Padding
• Activation function
• Regularization.

## Pooling
- Pooling layers are used to reduce the spatial dimensions of the input volume
![alt text](imgs/image-3.png)
- Min Pooling / Max Pooling / Average Pooling = Takes one of the segments and applies the function to it to pick the chosen value


## Full Convolutional Network
- Input
- Convolutional Layer 1
- Pooling Layer 1
- Convolutional Layer 2
- Pooling Layer 2
- Fully Connected Layer
- Output


## Convolutional Layers
### Action
- Apply filters to extract features from the input
- The filters are composed of small kernels which are learned 
- One bias term per filter
- Applying the activation function on every value of feature map


## Why do we use log loss?
- When the output is 0, the log loss is infinity
- When the output is 1, the log loss is 0

## What do CNN layers learn?
- The first layer learns simple features and basic details
- The middle layers learn more complex features and parts of the image (eyes, nose, etc.)
- The last layers learn to recognize full objects in different orientations and shapes

## Backpropagation
- Tells us how to adjust the weights in order to minimize the error of the network and improve the predictions
- The weights are adjusted in the opposite direction of the gradient of the loss function

## Layer's Reception Field
- The receptive field is the region in the input space 
that a particular CNN’s feature is looking at 
(i.e. is affected by).
- A layer's receptive field refers to the spatial region in the input data that influences the activations of neurons within that layer, typically expanding with each layer in a convolutional neural network as a result of successive convolution and pooling operations.

## Loss Function
### L1 distance
- Absolute difference between the true value and the predicted value
- L1 loss is less sensitive to outliers
### L2 distance
- Squared difference between the true value and the predicted value
- L2 loss is showing the outliers more than L1 loss

## Optimization 
- Partial Derivatives: The derivative of a function with respect to one of its variables, with the others held constant
-  SGD: Stochastic Gradient Descent
-  Adam: Adaptive Moment Estimation 

## Softmax
- Used for multi-class classification
- It converts the output to a range of 0 to 1
- The sum of the outputs is equal to 1
- Used to calculate the relative probabilities of the classes
- The one with the highest rate is the predicted class

- Works on the formula:
    - y = e^x / sum(e^x)
    - e^x is the exponential of x
    - sum(e^x) is the sum of the exponentials of all the classes
    - y is the probability of the class
    - x is the output of the network


## Multi label vs Multi class
- Multi label: Each sample can belong to multiple classes
- Multi class: Each sample belongs to one and only one class
- Example: 
    - Multi label: Car can be red, fast, and expensive
    - Multi class: Car can be red, blue, or green

## How to prevent overfitting
- Regularization
- Early stopping

## Batch Normalization
- Batch: A set of samples that are processed together
- Normalization: Scaling the input to have a mean of 0 and a standard deviation of 1
- Normalization is done for each feature in the input

## Gradient Accumulation
- Collects the gradients from multiple batches before updating the weights

## Model distillation
- Training a smaller model to mimic the predictions of a larger model
- The smaller model is trained on the outputs of the larger model
- The smaller network generalizes better than the larger network and is faster to train




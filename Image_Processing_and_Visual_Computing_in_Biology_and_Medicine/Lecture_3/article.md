# Article about the lecture: 

## Title: Neural Networks - Deep Learning

## Introduction
- Neural Networks are the foundation of deep learning popularized by the rise of big data and computational power. They are used in a wide range of applications such as computer vision, natural language processing, and speech recognition. This article will dive into the concepts of neural networks like: their fundamental building blocks, the architecture, the error calculation, the optimization process or the activation functions.


## What are the fundamental building blocks of a Neural Network?
The Neuron. Neuron (Perceptron) is composed of a weights, a bias, and a non-linear activation function. The mathematical representation of a neuron is: 
    - y = f(w1x1 + w2x2 + ... + wnxn + b)
    - y is the output
    - f is the activation function
    - w1, w2, ..., wn are the weights
    - x1, x2, ..., xn are the inputs
    - b is the bias

## What is a Neural Network?
A neural network is a set of algorithms resembling the human brain. It is composed from a set of inputs, a set of layers, and a set of outputs. Each layer is composed of a set of neurons. The neurons are connected to each other by a set of weights. Each of these weights is representing a parameter that is learned from the data. Neural network is an unsupervised learning algorithm, which means that it learns from the data that is not labeled. This means that the network is learning to find patterns in the data by itself instead of being told what to look for. This leads to the network being "creative" and finding patterns that are not obvious to the human eye. This feature of being able to find complex patterns in the data is what makes neural networks so powerful however, it also makes them hard to interpret and compute. 

### Architecture of the Neural Network
The architecture of a neural network is composed of three main layers: 
    - Input Layer
    - Hidden Layer
    - Output Layer

As can be seen on the listed figure each of the layers are interconnected with each other via a set of weights. The input layer is the first layer of the network and it is composed of a set of features. The hidden layer is the layer that is responsible for learning the patterns in the data. The output layer is the layer that is responsible for making the predictions. The number of inputs is defined by the number of features in the data. The number of outputs is defined by the number of classes in the data which we want to predict. Each of the neurons inside the hidden layer and output layer have its own activation function calculating the output of the neuron. 

## Activation Functions
The activation function as already mentioned is a non-linear function that is used to transform the input to the output. 

The most commonly used activation functions are: 
    - ReLU: Rectified Linear Unit
    - Sigmoid
    - Softmax

Each of these activation functions have their own use case and their own advantages and disadvantages. ReLU is the most commonly used activation function. It is used to capture the non-linear relationships in the data. Sigmoid is used for binary classification. Sigmoid is heavily overused and the fact it can cause the vanishing gradient problem is completely ignored. The vanishing gradient problem is a problem that occurs when the gradient of the loss function becomes very small meaning that the network has a very minimal chance of learning anything. Softmax is used for multi-class classification. It works on the premise of normalizing the output to a range of 0 to 1. The total sum of the outputs always equals 1 as it represents the "probability" of the class. This "probability" shall not be confused with the actual probability determined by the data. The use of the word "probability" is used to represent the relative probabilities of the classes.


## Error Calculation
The main concept of the error calculation is to calculate the distance between the network's output and the true output. Each type of problem has a different error calculation. For example, for regression problems the Mean Squared Error (MSE) is used. For binary classification problems the Binary Cross Entropy is used. For multi-class classification problems the Categorical Cross Entropy Loss is used. For multi-label classification problems the Multilabel margin loss is used. For segmentation problems the Dice Coefficient is used. For segmentation problems the Jaccard Index is used.



## Convolutional Neural Networks (CNN)
Convolutional Neural Networks are used for image data. They are composed of a set of convolutional layers and pooling layers. The convolutional layers are used to extract features from the image. The layers are not fully connected. Filters are used to extract features from the image such as edges, textures, etc. Each 2D slice of the filters are called a kernel. Convolutional layers are followed by pooling layers

---------------

image

---------------

As can be seen on the listed figure the Convolutional Network is combined of set of different layers. The input layer is the first layer of the network. The rule of thumb is that the input size is equal to the number of pixels in the image. The following layer is the convolutional layer. The convolutional layer is responsible for applying filters to the input to extract features from the image. The output of the convolutional layer is called a feature map. The number of feature maps is equal to the number of filters used in the convolutional layer. These feature maps are storing the features that are extracted from the image. The next layer is the pooling layer. The pooling layer can be compared to the activation function in the sense that it is responsible for modifying the output of the previous layer in a constant fashion. The availabe pooling functions are: Min Pooling, Max Pooling, and Average Pooling. The pooling layer is used to reduce the spatial dimensions of the input volume. The output of the pooling layer is a reduced version of the input volume. The last layer is the fully connected layer. The fully connected layer is responsible for actually making the predictions. The fully connected layer is connected to the output layer. Based on the output of the fully connected layer the network is assigning the class to the image.

### Padding
Concept of padding comes from the idea of preserving the spatial dimensions of the input volume as by default the convolutional layer is reducing the spatial dimensions of the input volume by applying the filter to the input which changes the dimensions of the input. The main premise of the padding is to add zeros to the input volume so that the output volume has the same dimensions as the input volume. This allows us to have the same dimensions of the input and the output volume.
- Full padding adds a border of zeros such taht all image pixels are visited the same number of times by the filter. Increasing the padding increases the size of the output volume. 
- Same padding adds a border of zeros such that the output volume has the same dimensions as the input volume.

### Stride
The stride is a "Measuring Window" size. It controls the magnitude of how much is the respective filter moved each time. The stride is used to control the output size of the volume. This is based on the principle that the output size is calculated by the 
formula: n(out) = (n(in) - 2p - k)/s + 1. 
The n(out) is the output size
n(in) is the input size
p is the padding
k is the filter size
s is the stride


### Convolutional Layers
The Actions:
    - Apply filters to extract features from the input
    - The filters are composed of small kernels which are learned
    - One bias term per filter
    - Applying the activation function on every value of feature map

Parameters:
    - Number of Kernels
    - Kernel Size D, H, W is defined by the input tensor
    - Stride
    - Padding
    - Regularization

Input & Output:
    - Input: 3D tensor: Width × Height × Channels 
    - Output: 3D tensor: Width × Height × Feature Maps (one for each filter)
            : 2D Feature Map per filter

### Pooling
The Actions:
    -  Reduce the spatial dimensions of the input volume
    - Sliding a window over the input and applying a function to it

Parameters:
    - Pooling Type: Min, Max, Average
    - Pooling Size: D, H, W
    - Stride
    - Padding

Input & Output:
    - Input: 3D tensor: Width × Height × Channels (or Feature Maps)
    - Output: 3D tensor: Width × Height × Channels (or Feature Maps)
            : 2D Feature Map per filter 
            : Reduced spatial dimensions

### Full Convolutional Network
The Action:
    - Collect the information from the final feature map
    - Generate the final output (prediction)

The Parameters:
    - Number of Nodes
    - Activation Function

Input & Output:
    - Input: Flattened 3D tensor 
    - Output: 3D tensor: Width × Height × Channels
            : 2D Feature Map per filter 


## Success evolution through the layers
The evolution of the outcome is improved over the hidden layers. There would not be a significant reason to keep extending the number internal layers if the network is not learning anything new. The first layer learns simple features and basic details. The middle layers learn more complex features and parts of the image (eyes, nose, etc.). The last layers learn to recognize full objects in different orientations and shapes.
This evolution is the reason why the convolutional neural networks are so powerful. They are able to propagate the information from the input to the output in a way that the network is able to learn the patterns in the data.

## Backpropagation
In order to imrove over time the Neural Network use a concept of Backpropagation. The backpropagation is a process that tells us how to adjust the weights in order to minimize the error of the network and improve the predictions 

---------------

image

---------------

The next figure is showing us a real example on how the backpropagation is working. The data we are working with is the output of convolutional layer to the pooling layer using the max function. After successfully calculating the output of the pooling layer. We can see the error that is calculated by the network using backpropagation. Even tough we used the Max Pooling function the error is still calculated in the same way as it would be calculated if we used the Min Pooling or Average Pooling function. The magnitude of the error is represented with the color blue and the actual result of the Max Pooling function with a small red number in each of the cells of the output. This means that in order for us to get the best results the number calculated by the Max Pooling function should be as close as possible to the actual result and therefore our error should be as small as possible. Important thing to be noted is that backpropagation is used to calculate the error and not to calculate the output of the network. Its usage is also not a one-time thing. The backpropagation is used to calculate the error of the network every single time the network is making a prediction. This is the reason why the backpropagation is so important. The continuous usage of the backpropagation is what makes the network learn from the data and improve over time. 

## Model Distillation
The idea of having large models is a thrilling one. However, the large models are not always the best solution and often are not sustainable and achievable unless the computational power is not a problem. The large models are often slow to train and hard to interpret. They might introduce a lot of noise and potentially have learned occasional patterns that are not relevant to the test data. The model distillation is a process of creating a smaller model which is trained to mimic the predictions of a larger model. The key concept of the model distillation is that the smaller model is trained on the outputs of the larger model. This allows the smaller model to be trained very efficiently with already pre-processed data and observed patterns. The smaller model is able to generalize better due to the lack of noise and irrelevant patterns. The trainig time of smaller model is a fraction of the time needed to train the larger model. This allows the Machine Learning Engineers to save time, resources,  and observe new behvaiours of the model which is not influenced by the noise and irrelevant patterns.

## Optimization
Purpose of the optimization is to minimize the error of the network and improve the predictions. The optimization is done by adjusting the weights of the network. The most commonly used optimization algorithms are: 
    - Gradient Descent
    - Stochastic Gradient Descent
    - Adam
Partial derivatives are used to calculate the gradient of the loss function. The gradient is then used to adjust the weights of the network with the goal of minimizing the error of the network. The learning rate is a hyperparameter that is used to control the magnitude of the weight updates.
The crucial aspect to look at is to not set the learning rate too high as the model is now being able to discover local minima. On the other hand when learning rate is too small the model gets stuck in the local minima not being able to discover more and plateus.

## Conclusion
This Article solves the problem of understanding the Neural Networks and Convolutional Neural Networks. It provides a comprehensive overview of the fundamental building blocks of the Neural Networks, the architecture, the error calculation, the optimization process, the activation functions. The insights provided in this article are based on the real-world experience and in-depth research. The article is a great starting point for anyone who wants to learn more about Neural Networks and Convolutional Neural Networks.

At the conclusion of the article, I would like to thank Mr. Liad Magen for his guidance and provision of materials explaining the discussed topics. His expertise and support have been instrumental in enriching the content and insights presented.
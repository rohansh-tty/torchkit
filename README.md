# torchkit

![](https://github.com/Gilf641/EVA-6/blob/master/torchkit/torchkit.png)

PyTorch Utility Package to setup training and testing pipeline for Computer Vision Tasks

## File Structure

![](https://github.com/Gilf641/EVA-6/blob/master/torchkit/treestructure.png)

Package has 5 sub-packages

### 1. data
Consists of Dataset, Dataloader functions and classes

### 2. models 
Has two different network files, based on CIFAR-10 and MNIST

### 3. run
Consists of Train and Testing part of NeuralNet

### 4. torchsummary
Mainly modelsummary with Receptive Field calculated layer-wise

### 5. utils
Consists of DataUtils and ModelUtils, which has helper functions mainly to plot and visualize data in former, & latter has model related functions.


## Features

#### Convolutions
    * Depthwise
    * Dilated 

#### Normalization
    * BatchNorm
    * GroupNorm
    * LayerNorm


#### Model Summary

    * with layer-wise Receptive Field

#### Model utilities

    Loss functions

        * Cross Entropy Loss
        * NLLoss

    Evaluation Metrics

        * Accuracy

    Optimizers

        * Stochastic Gradient Descent

    LR Schedulers

        * Step LR
        * Reduce LR on Plateau
        * One Cycle Policy



    


#### Datasets

    * MNIST
    * CIFAR10
  



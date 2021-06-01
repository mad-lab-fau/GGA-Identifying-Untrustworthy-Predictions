# GGA-Identifying-Untrustworthy-Predictions
Code relative to "Identifying Untrustworthy Predictions in Neural Networks by Geometric Gradient Analysis"
*Leo Schwinn, An Nguyen, René Raab, Leon Bungert, Daniel Tenbrinck, Dario Zanca, Martin Burger, Bjoern Eskofier*
Paper: https://arxiv.org/abs/2102.12196
Accepted at UAI 2021

We propose a geometric gradient analysis (GGA) of the input gradients of neural networks to detect out-of-distribution data and adversarial attacks. GGA does not require retraining of a given model. Here, we analyze and interpret the gradient of a neural network w.r.t. its input (e.g., saliency map). More precisely, for a given input sample we inspect the geometric relation among all possible saliency maps, calculated for each output class of the model. This is achieved by a pairwise calculation of the cosine similarity between saliency maps. The cosine similarites for a given input can be summarizes with cosine similiarty maps (CSMs)

The following image exemplifies how GGA can be used to differentiate different data types for a MNIST model by calculating the respective CSM for every input.

<img src="./Images/CSM_MNIST.JPG">

Examples of CSMs for the "Standard" CIFAR10 model provided by RobustBench libary (https://github.com/RobustBench/robustbench) are given below:

<p float="left">
  <img src="./Images/CSM_Clean_Data_CIFAR10_Model.png", width=400>
  <img src="./Images/CSM_Noisy_Data_CIFAR10_Model.png", width=400>
</p>


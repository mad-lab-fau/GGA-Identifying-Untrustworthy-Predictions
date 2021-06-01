# GGA-Identifying-Untrustworthy-Predictions
Code relative to "Identifying Untrustworthy Predictions in Neural Networks by Geometric Gradient Analysis"
*Leo Schwinn, An Nguyen, René Raab, Leon Bungert, Daniel Tenbrinck, Dario Zanca, Martin Burger, Bjoern Eskofier*
Paper: https://arxiv.org/abs/2102.12196
Accepted at UAI 2021

We propose a geometric gradient analysis (GGA) of the input gradients of neural networks to detect out-of-distribution data and adversarial attacks. GGA does not require retraining of a given model.

The following image exemplifies how GGA can be used to differentiate different data types for a MNIST model.

<img src="./Images/CSM_MNIST.JPG">

Examples of the "Standard" CIFAR10 model provided by robust bench are given below:

<img src="./Images/CSM_Clean_Data_CIFAR10_Model.png">
<img src="./Images/CSM_Noisy_Data_CIFAR10_Model.png">

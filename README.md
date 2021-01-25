# Multilayer-Perceptron-Classifier
## CECS 550 Project 2 — MLP

### Introduction: 
   1. MLP-Classifier.py and MLP-Classifier.ipnyb are same files with different extension and have the same content. 
   2. Our code can be run on either console or on JupyterNotebook.
   3. The file, MLP.html is a snapshot of our entire code along with output. This file is just for reference.
   4. We have used artificial neural network concept for this project.
   5. The code has ANN class, it contains the following methods.
   
      i. The class constructor initiates weights with random numbers.
      
      ii. It has a feed forward propogation method. For the hidden layer we are using "sigmoid" as an activation function, and for the output layer we are using "hyperbolic    tangent" as activation function.
      
      iii. It has a back propogation method. Gradient decent is implemented for updating weights with learning rate.      
      Learning rate is not a constant, it decays over time using the formula "lr(t)=initial_lr * e^(-alpha * t)".
      
      iv. We have a prediction method to predict the output of the testing set using the trained neural network.
      
      v. We have an accuracy method to calculate the accuracy.
      
      vi. We have sigmoid method, MSE method, and more_forgiving method.
      
   6. We were able to achieve an accuracy of 40% for the testing dataset.
   7. MLP Architecture will be generated in a seperate window when the code is executed.You will have to close that window for the rest of the code to run.
    We used eiffel2 package for this.

### Steps to run on Console:
   i. Open command prompt and navigate to the folder that has MLP-Classifier.py and sample files( 550-p1-cset-krrk-1.csv, 550-p1-csect-krrk-2.csv).
   
   ii. Make sure that MLP-Classifier.py and samplefiles are in the same folder.
   
   iii. Run "python MLP-Classifier.py" command in the command prompt.
 
### Steps to run on JupyterNotebook:
   i. Open Anaconda prompt.
   
  ii. Enter command "jupyter notebook".
  
  iii. The jupyter notebook will run on localhost:8888
  
  iv. Navigate through jupyternotebook and open D-Tree.ipynb
  
  v. Select Kernel option in the navigation bar and select "restart and run all".

### Contents:  
MLP-Classifier.py, MLP-Classifier.ipynb, Dataset.txt, MLP.html, MLP.jpeg.

### Setup and Installation: 
   1. System should have python installed.
   2. Should install libraries- numpy, random, math, re, eiffel2.
   3. Should install anaconda, JupyterNotebook( Optional): Instead of running our program on console, you can use anaconda -JupyterNotebook.

### Sample invocation:  
Samples - Dataset.txt was read into the code using open() and text from the dataset was formatted using regular expression (import re).

### References:
1. Assignments("Artificial Neural network from scratch to classify an XOR gate") from CECS 550 Advanced AI by professor J.Moon

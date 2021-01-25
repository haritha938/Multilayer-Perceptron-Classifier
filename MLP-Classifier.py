# # Project 2 — MLP Classifier
# 
# Team Name: JAKH
# 
# Author:
#  1. Jaya Krishna Kalavakuri (026374853)
#     Contact : JayaKrishna.Kalavakuri@student.csulb.edu
#  2. Akhil Varupula (025534780)
#     Contact : Akhil.Varupula@student.csulb.edu
#  3. Kiran Panjam (026642549)
#     Contact : Kiran.Panjam01@student.csulb.edu
#  4. Haritha Nimmagadda (026636140)
#     Contact : Haritha.Nimmagadda@student.csulb.edu


#Imports
import numpy as np
import re
import random
import math
from eiffel2 import builder


Dataset = [] #Dataset
X_id = [] #Input id
X_input = [] #Input data
y_label = [] #class

f = open("Dataset.txt", "r")    #Opening Dataset.txt to read it's contents
for x in f:
    data = re.sub(r'[()]', '', x)    #Reading Data with "(" and ")"
    data = data.split()
    Dataset.append(data)
random.shuffle(Dataset)            #Shuffling Data
for data in Dataset:
    X_id.append(int(data[0])) #sepqrating data by column into three different variable.
    num = []
    for i in data[1:len(data) - 1]:
        num.append(int(i))
    X_input.append(num)
    y_label.append(int(data[len(data) - 1]))


X = np.array(X_input)
X = X.astype(np.float)     #Converting lists into arrays
y_val = np.array(y_label)


if (np.isnan(X).any()):          #Checking for NaN values in the Input Data.
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            if (np.isnan(X[i][j])):
                X[i][j] = random.randint(np.nanmin(X), np.nanmax(X))


def normalize_data(X):               #Normalizing Input Data to range[0,1]
    for i in range(np.shape(X)[1]):
        max_ = max(X[:,i])
        min_ = min(X[:,i])
        for j in range(np.shape(X)[0]):
            X[j][i] = (X[j][i] - min_)/(max_ - min_)
    return X

X = normalize_data(X)
X[:5]                  #Normalized Input values


if (np.isnan(y_val).any()):         #Checking for NaN values in Class values
    for i in range(np.shape(y_val)[0]):
        if (np.isnan(y_val[i])):
            y_val[i] = random.randint(np.nanmin(y_val), np.nanmax(y_val))


y = np.zeros((np.size(y_val), np.size(np.unique(y_val)))) #Obtaining features of Class values
for i in range(np.shape(y)[0]):
    y[i][y_val[i]] = 1


for i in range(10):       #Class Values and Features
    print(y[i], y_val[i])


def train_test_split(X, y, y_val, ratio):   #Train test split function
    split = (len(X) * ratio) // 100
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
    y_val_train = y_val[:split]
    y_val_test = y_val[split:]
    
    return X_train, X_test, y_train, y_test, y_val_train, y_val_test


X_train, X_testing, y_train, y_testing, y_val_train, y_val_testing = train_test_split(X, y, y_val, 80)           #Calling Train Test Split function
X_test, X_hold, y_test, y_hold, y_val_test, y_val_hold = train_test_split(X_testing, y_testing, y_val_testing, 50)


class ANN:        #MLP Classifier class
    def __init__(self, X, y_val, Hidden_nodes, lr):            #Initial Function
        print("MLP Architecture:\n")
        builder([np.shape(X)[1], Hidden_nodes, np.size(np.unique(y_val))]) #Architecture
        self.weights_1 = np.random.randn(np.shape(X)[1], Hidden_nodes)
        print("\nInitial Weights 1:\n", self.weights_1)
        self.weights_2 = np.random.randn(Hidden_nodes, np.size(np.unique(y_val)))
        print("\nInitial Weights 2:\n", self.weights_2)
        
        self.lr = [lr]
        self.lr_0 = lr
        self.error = []
        
        
    def sigmoid(self, z):            #Sigmoid Function.
        return 1/(1 + np.exp(-z))
    
    def MSE(self, y):                #Mean Squared Error Function
        mse = 0
        mse = np.sum(np.square(self.y_c - y))/(y.size * 2)
        
        return mse
    
    def model_fit(self, X, y, y_val, X_h, y_h, y_val_h, epochs, B_size):    #Model Fit
        alpha = 1 / epochs           #alpha for Learning rate decay
        min_er = 1.0
        self.hold_acc = []
        self.train_acc = []
        for i in range(epochs):
            for j in range(0, np.shape(X)[0], B_size):
                y_temp = self.forward_propogation(X[j: j+B_size])      #Forward Propogation Function
                self.backward_propogation(X[j:j+B_size], y[j: j+B_size])     #Backward Propogation Function
                if j % 10 == 0:
                    temp = self.MSE(y[j: j+B_size])
                    if temp < min_er:
                        optimal_w1 = self.weights_1
                        optimal_w2 = self.weights_2
                    self.error.append(temp)
                    ind = len(self.error)
                    self.lr.append(self.lr_0 / math.exp(alpha * i))           #Learning rate decay with constant alpha
            self.prediction(X_h, y_h, optimal_w1, optimal_w2)                 #
            self.prediction(X, y, optimal_w1, optimal_w2)                     #Predicting and
            h_acc = self.accuracy(y_val_h)                                    #Calculating accuracy score 
            t_acc = self.accuracy(y_val)                                      #for Training and Holdout set
            self.hold_acc.append(h_acc)                                       #for every epoch for comparision
            self.train_acc.append(t_acc)                                      #
            print("Epoch:{:5d}".format(i + 1), end = " ")                     #
            print("Training accuracy:{:2.2f}".format(t_acc), end = " ")       #
            print("Holdout accuracy: {:2.2f}".format(h_acc))                  #
        return optimal_w1, optimal_w2
                    
        
    def forward_propogation(self, X):                       #Forward Propogation Function
        self.a_1 = np.array(np.dot(X, self.weights_1))
        self.h_1 = np.array(self.sigmoid(self.a_1))         #Activation Sigmoid
        self.a_2 = np.array(np.dot(self.h_1, self.weights_2))
        self.h_2 = np.array(np.tanh(self.a_2))              #Activation Hyperbolic Tangent
        self.y_c = self.h_2
        return self.y_c
    
    def backward_propogation(self, X, y):                  #Backward Propogation Function
        
        Dweights_2 = np.zeros(np.shape(self.weights_2))
        
        temp = np.multiply((1 - np.square(self.y_c)), (self.y_c - y))
        Dweights_2 = np.dot(self.h_1.T, temp)              #Partial Derivative formula for Weights 2
        
        Dweights_1 = np.zeros(np.shape(self.weights_1))
                                                           #Partial Derivative formula for Weights 1
        temp = np.dot(self.weights_2, np.multiply((self.y_c - y), np.multiply((1 - self.y_c), self.y_c)).T)
        Dweights_1 = np.dot(X.T, np.multiply(temp.T, np.multiply(self.h_1, (1 - self.h_1))))
        
        ind = len(self.lr) - 1
        self.weights_2 = self.weights_2 - (self.lr[ind] * Dweights_2)   #Gradient decent updating Weights
        self.weights_1 = self.weights_1 - (self.lr[ind] * Dweights_1)
        
    def more_forgiving(self, y):            #More Forgiving (1 if >= 0.8)
        for i in range(np.size(y)):
            if (y[i] >= 0.8):
                y[i] = 1.0
            elif (y[i] <= 0.2):
                y[i] = 0.0
            else:
                y[i] = 0.5
        return y
    
    def prediction(self, X, y, op_w1, op_w2):      #Prediction function
        self.y_pred = np.zeros(np.shape(y))
        for i in range(np.shape(X)[0]):
            # Forward Propogation for Test input with optimal weights.
            a_1 = np.array(np.dot(X[i], op_w1))
            h_1 = np.array(self.sigmoid(a_1))
            a_2 = np.array(np.dot(h_1, op_w2))
            h_2 = np.array(np.tanh(a_2))
            y_res = h_2
            y_res = np.squeeze(y_res)
            self.y_pred[i] = self.more_forgiving(y_res)    #Calling More Forgiving function
    
    def accuracy(self, y_val):         #Accuracy Calculator
        count = 0
        acc = 0
        for i in range(np.size(y_val)):
            pred = -1
            y_list = list(self.y_pred[i])
            for j in range(len(y_list)):
                if (y_list[j] == 1):
                    pred = j
            if (y_val[i] == pred):
                count += 1
        acc = (count / np.size(y_val)) * 100
        return acc


jakh = ANN(X_train, y_val_train, 12, 0.1)  #Assigning Class
print("\nTraining.....")
w1, w2 = jakh.model_fit(X_train, y_train, y_val_train, X_hold, y_hold, y_val_hold, 30, 10)
jakh.prediction(X_test, y_test, w1, w2)
print("\nFinal Weights 1:\n", w1)
print("\nFinal Weights 2:\n", w2)
print("\nAccuracy Testing Data:{:4.2f}".format(jakh.accuracy(y_val_test)))

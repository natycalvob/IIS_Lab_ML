#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Intelligent and Interactive Systems 
Spring 2019
Lab 2: Software for Machine Learning 
"""
# Import datasets, classifiers and performance metrics

import numpy as np
import cv2

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import manifold, neighbors, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


'''
Global Variables 
'''
SIMPLE_EMBEDDING=False
MANUAL_SPLIT    =True
DATA_2D         =True


'''
Functions 
'''
def displayImages(digitsIm):
    '''
    This fuction allows viweing the data
    '''
    i = 0
    for image in digitsIm:
        if(i < 10):
            imMax = np.max(image)
            image = 255*(np.abs(imMax-image)/imMax)
            res = cv2.resize(image, (100, 100), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite('digit_'+str(i)+'.png',res)
            i+=1
        else:
             break

    
def holdOut(fnDigits, fnData, nSamples, percentSplit=0.8):
    '''
    This function splits the data into training and test sets
    '''
    if(MANUAL_SPLIT):
        n_trainSamples = int(nSamples*percentSplit)
        trainData = fnData[:n_trainSamples,:]
        trainLabels = fnDigits.target[:n_trainSamples]
        
        testData = fnData[n_trainSamples:,:]
        expectedLabels = fnDigits.target[n_trainSamples:]
    else:
        trainData, testData, trainLabels, expectedLabels = train_test_split(fnData, fnDigits.target,
                                                                            test_size=(1.0-percentSplit), random_state=0)

    return trainData, trainLabels, testData, expectedLabels


def plotData(X):
    '''
    This function plots either 2D data or 3D data
    '''
    if(DATA_2D):
        plt.scatter(X[:,0], X[:,1])
        plt.show()
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X[:,0], X[:,1], X[:,2])
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.show()

'''
Main Program 
'''
# Data Visualization 
digits = datasets.load_digits()
print type(digits)
displayImages(digits.images)


# Transforming the data in a (samples, features) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Data Dimension
if(DATA_2D):
    nComp = 2
else:
    nComp = 3

# Data Reduction 
if(SIMPLE_EMBEDDING):
    #  Dimensionality Reduction PCA 
    pca = PCA(n_components = 2)
    X_trans = pca.fit_transform(data)
else:
    # Manifold embedding with tSNE
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_trans = tsne.fit_transform(data)
    
# Ploting Data
plotData(X_trans)

# Manually Split your data
X_train, X_labels, X_test, X_trueLabels = holdOut(digits, data, n_samples)
print(data.shape)
print(X_labels.shape)
print(X_trueLabels.shape)

n_neighbors = 10
# k-NearestNeighbour Classifier 
kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")

# Training the model
kNNClassifier.fit(X_train, X_labels)
predictedLabels = kNNClassifier.predict(X_test)

# Display classifier results
print("Classification report for classifier %s:\n%s\n" 
      % ('k-NearestNeighbour', metrics.classification_report(X_trueLabels, predictedLabels)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(X_trueLabels, predictedLabels))

# Cross Validation 
scores = cross_val_score(kNNClassifier, data, digits.target, cv=5)
print(scores)

# Support Vector Machines
clf_svm = LinearSVC()

# Training 
clf_svm.fit(X_train, X_labels)

# Prediction
y_pred_svm = clf_svm.predict(X_test)
acc_svm = metrics.accuracy_score(X_trueLabels,y_pred_svm)

print "Linear SVM accuracy: ",acc_svm

# Cross Validation
scores = cross_val_score(clf_svm, data, digits.target, cv=5)
print(scores)
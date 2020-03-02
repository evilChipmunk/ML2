from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy import array
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold 
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support

from sklearn.neural_network import MLPClassifier
   
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
 
import util
import data
import plotter
import searcher
import sklearn


def adultFit(dataType):
  
 
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
 
    xLabel = 'Network Layers'
    scoreList = util.ScoreList(xLabel) 
    title =  '{0} Neural Network Fit Times'.format(dataType)
  
    params = {'activation': 'relu', 'learning_rate': 'invscaling', 'solver': 'lbfgs'} 
 
 
    input = package.features.shape[1]
    input = int(.7 * input) 
 
    param_range = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 400, 500, 600, 700, 800]
 
    times = []
    for param in param_range:

        clf = MLPClassifier(hidden_layer_sizes = (input,7,2)) 
        clf.set_params(**params)   
        clf.max_iter = param
        start = time()
        clf.fit(xTrain, yTrain)
        end = time()
        times.append(end-start)
    
    plotter.plot(param_range, times, 'max_iter', 'fit times', title)  
    return

 
 

def heartFit(dataType):
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
 
    xLabel = 'Network Layers'
    scoreList = util.ScoreList(xLabel) 
    title =  '{0} Neural Network Fit Times'.format(dataType)
  
    params = {'activation': 'tanh', 'learning_rate': 'constant', 'solver': 'adam'}
 
 
    input = package.features.shape[1]
    input = int(.7 * input) 
 
    param_range = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 400, 500, 600, 700, 800]
 
    times = []
    for param in param_range:

        clf = MLPClassifier(hidden_layer_sizes = (input,5,2)) 
        clf.set_params(**params)   
        clf.max_iter = param
        start = time()
        clf.fit(xTrain, yTrain)
        end = time()
        times.append(end-start)
    
    plotter.plot(param_range, times, 'max_iter', 'fit times', title)  
    return
 


def adult(dataType):
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
 
    xLabel = 'Network Layers'
    scoreList = util.ScoreList(xLabel) 
    title =  '{0} Neural Network'.format(dataType)
 
    params = {'activation': 'relu', 'learning_rate': 'adaptive', 'solver': 'sgd'}
    params = {'activation': 'relu', 'learning_rate': 'invscaling', 'solver': 'lbfgs'} 
    # params = searcher.searchNetwork(xTrain, yTrain, xTest, yTest)

    clf = MLPClassifier(max_iter=250)
    input = package.features.shape[1]
    input = int(.7 * input) 
    # hiddenLayers = (input, 20)
    # hiddenLayers = (input,)
    # param_range = [(input,1,2), (input,2,2),(input,3,2),(input,4,2),(input,5,2),(input,6,2),(input,7,2),(input,8,2),(input,9,2), (input,10,2)]
    # xRange = [1, 2, 3, 4, 5, 6, 7 , 8, 9, 10]
    # plotter.plotValidationCurve(clf, xTrain, yTrain, 'hidden_layer_sizes', param_range, title + ' Hidden Layers ', xRange)

    # clf = MLPClassifier(hidden_layer_sizes = (input,7,2)) 
    # clf.set_params(**params) 
    # param_range = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 400, 500, 600, 700, 800]
    # plotter.plotValidationCurve(clf, xTrain, yTrain, 'max_iter', param_range, graphTitle=title + ' Max Iterations ')

    clf = MLPClassifier(hidden_layer_sizes = (input,7,2)) 
    clf.max_iter = 150
    plotter.plotLearningCurves(clf, title=title, xTrain=xTrain, yTrain=yTrain)
    title = 'Adult' 
    clf.fit(xTrain, yTrain)
    plotter.plotConfusion(clf, title, ['>50K', '<=50K'], xTest, yTest)
    return
 

def heart(dataType):
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
 
    xLabel = 'Network Layers'
    scoreList = util.ScoreList(xLabel) 
    title =  '{0} Neural Network'.format(dataType)
  
    params = {'activation': 'tanh', 'learning_rate': 'constant', 'solver': 'adam'}
    # params = searcher.searchNetwork(xTrain, yTrain, xTest, yTest)

    clf = MLPClassifier(max_iter=600)
    clf.set_params(**params)
    input = package.features.shape[1]
    input = int(.7 * input) 
    # hiddenLayers = (input, 20)
    # hiddenLayers = (input,)
    # param_range = [(input,1,2), (input,2,2),(input,3,2),(input,4,2),(input,5,2),(input,6,2),(input,7,2),(input,8,2),(input,9,2), (input,10,2)]
    # xRange = [1, 2, 3, 4, 5, 6, 7 , 8, 9, 10]
    # plotter.plotValidationCurve(clf, xTrain, yTrain, 'hidden_layer_sizes', param_range, title + ' Hidden Layers ', xRange)

    # clf = MLPClassifier(hidden_layer_sizes = (input,5,2)) 
    # clf.set_params(**params) 
    # param_range = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 400, 500, 600, 700, 800]
    # plotter.plotValidationCurve(clf, xTrain, yTrain, 'max_iter', param_range, graphTitle=title + ' Max Iterations ')

    clf = MLPClassifier(hidden_layer_sizes = (input,5,2)) 
    clf.set_params(**params)   
    clf.max_iter = 600
    plotter.plotLearningCurves(clf, title=title, xTrain=xTrain, yTrain=yTrain)
    title = 'Heart' 
    clf.fit(xTrain, yTrain)
    plotter.plotConfusion(clf, title, ['Diameter narrowing ', 'Diameter not narrowing'], xTest, yTest)
    return

 

import warnings
warnings.filterwarnings("ignore")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
 

import numpy as np 
import mlrose
import util
from sklearn.preprocessing import StandardScaler
  
from numpy import array
import pandas as pd
from time import time 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold 
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support

from sklearn.neural_network import MLPClassifier
# from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold
   
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
 
 
import data
import plotter
import searcher
import sklearn
 
from sklearn.model_selection import cross_val_score

 
def mlRoseAdult(package, iterations):
     
    input = package.features.shape[1]
    input = int(.7 * input)   

  
    # clf = mlrose.neural.NeuralNetwork(hidden_nodes=[input, 7, 2],
    #                         algorithm='gradient_descent',
    #                         activation='relu',
    #                         bias=True, is_classifier=True, learning_rate=0.001,
    #                         early_stopping=True, clip_max=1)
    # plotNetwork(clf, iterations, 'Adult - Gradient Descent', package)
     

    clf = mlrose.neural.NeuralNetwork(hidden_nodes=[input, 7, 2],
                            algorithm='random_hill_climb',
                            activation='relu',
                            bias=True, is_classifier=True, learning_rate=0.001,
                            early_stopping=True, clip_max=1)
    plotNetwork(clf, iterations, 'Adult - Random Hill Climb', package)
      
    clf = mlrose.neural.NeuralNetwork(hidden_nodes=[input, 7, 2],
                            algorithm='simulated_annealing',
                            activation='relu',
                            bias=True, is_classifier=True, learning_rate=0.001,
                            early_stopping=True, clip_max=1)
    plotNetwork(clf, iterations, 'Adult - Simulated Annealing', package)
      

    clf = mlrose.neural.NeuralNetwork(hidden_nodes=[input, 7, 2],
                            algorithm='genetic_alg',
                            activation='relu',
                            bias=True, is_classifier=True, learning_rate=0.001,
                            early_stopping=True, clip_max=1)
    plotNetwork(clf, iterations, 'Adult - Genetic Algorithm', package)
  

    return
    

def adult(package, iterations):  
    params = {'activation': 'relu', 'learning_rate': 'invscaling', 'solver': 'lbfgs'} 
 
    input = package.features.shape[1]
    input = int(.7 * input)  
    (input,7,2) 
    clf = MLPClassifier(hidden_layer_sizes = (input,7,2))
    clf.set_params(**params)
    plotNetwork(clf, iterations, 'Adult - Baseline', package)
 
    return 
 
 
def mlRoseHeart(package, iterations):
 

    input = package.features.shape[1]
    input = int(.7 * input)  
    (input,7,2)
  
    clf = mlrose.neural.NeuralNetwork(hidden_nodes=[input, 5, 2],
                            algorithm='gradient_descent', activation='relu'
                            , bias=True
                            , is_classifier=True #, learning_rate=0.001,
                            , early_stopping=True
                            , clip_max=5)
    plotNetwork(clf, iterations, 'Heart - Gradient Descent', package)
 
     
 
    clf = mlrose.neural.NeuralNetwork(hidden_nodes=[input, 5, 2],
                            algorithm='random_hill_climb',
                            activation='relu'
                            # , bias=True
                            , is_classifier=True 
                            #, learning_rate=0.001
                            #, early_stopping=True
                            # , clip_max=5
                            )
    plotNetwork(clf, iterations, 'Heart - Random Hill', package) 

 
    clf = mlrose.neural.NeuralNetwork(hidden_nodes=[input, 5, 2],
                            algorithm='simulated_annealing',
                            activation='tanh'
                            ,bias=True
                            , is_classifier=True
                            # , schedule=mlrose.ArithDecay()
                             #, learning_rate=0.001
                            # , early_stopping=True
                            # , clip_max=5
                            )
    plotNetwork(clf, iterations, 'Heart - Simulated Annealing', package) 
   
    clf = mlrose.neural.NeuralNetwork(hidden_nodes=[input, 5, 2],
                            algorithm='genetic_alg',
                            activation='tanh'
                            ,bias=True
                            , is_classifier=True
                            , learning_rate=0.001
                            , early_stopping=True
                            , pop_size=100
                            # , clip_max=5
                            )
    plotNetwork(clf, iterations, 'Heart - Genetic Algorithm', package)  
    return
    

def heart(package, iterations):  
 
    params = {'activation': 'tanh', 'learning_rate': 'constant', 'solver': 'adam'}
 
 
    input = package.features.shape[1]
    input = int(.7 * input) 
 
  
    clf = MLPClassifier(hidden_layer_sizes = (input,5,2))
    clf.set_params(**params)
    plotNetwork(clf, iterations, 'Heart Baseline', package)   

    return
 

def plotNetwork(clf, iterations, title, data):
    print()
    print(title)
    xTrain = data.xTrain
    xTest = data.xTest 
    yTrain = data.yTrain.values.ravel()
    yTest = data.yTest.values.ravel()

    x = []
    y = []  
    yT = []
    timeData = []
    devSingle = []
    for i in iterations: 
        s = time()
        clf.max_iters=i
        clf.max_iter=i 

        allscores = []
        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(xTrain, yTrain):
            crossXTrain, crossXTest = xTrain[train_index], xTrain[test_index]
            crossYTrain, crossYTest = yTrain[train_index], yTrain[test_index]


            clf.fit(crossXTrain, crossYTrain)
            cross_preds = clf.predict(crossXTest)
            prec, rec, f1, sup = precision_recall_fscore_support(crossYTest, cross_preds, average='macro')
            accScore = accuracy_score(crossYTest, cross_preds, True)
            
            
            REAL_TEST_PREDICTIONS = clf.predict(xTest)
            testprec, testrec, testf1, testsup = precision_recall_fscore_support(yTest, REAL_TEST_PREDICTIONS, average='macro')
            testaccScore = accuracy_score(yTest, REAL_TEST_PREDICTIONS, True)
            allscores.append([prec, rec, f1, accScore, testprec, testrec, testf1, testaccScore])
        e = time()
        # timeData.append(int(e - s))
        timeTaken = e - s
        timeData.append(timeTaken)

        allscores = np.array(allscores)
        f1 = allscores[:,2]
        testf1 = allscores[:,6]
        # accScore = cross_val_score(clf, X_train, y_train, n_jobs=1)
        mean = np.mean(f1)
        dev = np.std(f1)
        score = np.mean(testf1)
        print('Itt: {0} - Score: {1} - Time:{2}'.format(i, mean, timeTaken))

        devSingle.append(dev)
        x.append(i)
        y.append(mean)   
        yT.append(score)
        

    
    dev = np.std(y, axis=0)  
    dev = np.array(devSingle)
    fig, ax = plt.subplots()
    plt.title(title) 
    fig.tight_layout()
    ax.plot(x, y, label='Train Score')  
    ax.plot(x, yT, label='Test Score')  
    ax.fill_between(x, y - dev,  y + dev, alpha=0.1)

    # plt.legend(loc='best')
    color = 'tab:blue'
    ax.set_ylabel('Score', color=color)
    ax.set_xlabel('Iterations')
        
    ax2 = ax.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Time (s)', color=color)
    ax2.plot(x, timeData, color=color, alpha=0.3, label='Time')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)
    # plt.savefig('{0}\{1}.png'.format('C:\\Users\\mwest\\Desktop\\ML\\source\\Machine-Learning-Local\\Graphs\Two', title))
    plt.show()
    return
 

def run(dataType): 
    # dataType = 'heart'
    package = data.createData(dataType) 
    if dataType == 'heart':
        iterations = range(997, 1000)
        iterations = [100, 600, 997, 999, 1000, 2000, 3000, 4000]
        iterations = []
        for i in range(100, 4000, 20):
            iterations.append(i)
        # iterations = [599, 6000]
        heart(package, iterations)
        mlRoseHeart(package, iterations)
    # else:    
    #     iterations = range(1, 800)
    #     iterations = [799, 800]
    #     adult(package, iterations)
    #     mlRoseAdult(package, iterations) 
    return
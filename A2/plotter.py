import util

import numpy as np 
import pandas as pd
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report,confusion_matrix 
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support
 
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import ShuffleSplit

baseGraphPath = r'Graphs\\' 
showPlot = False

def getConfusionMatrix(yTest, predictions):
    print(confusion_matrix(yTest,predictions))
    print(classification_report(yTest,predictions))

def plotScores(scores, title, xLabel, yLabel='Score'): 

    x = []

    accuracy = []
    f1 = []
    prescion = []
    recall = []
    auc = []
    
    trainaccuracy = []
    trainf1 = []
    trainprescion = []
    trainrecall = []
    trainauc = []

    for score in scores:
        x.append(score.HyperParam)
        accuracy.append(score.Accuracy)
        prescion.append(score.Precision)
        recall.append(score.Recall)
        f1.append(score.F1)
        auc.append(score.AUC)
        
        trainaccuracy.append(score.TrainAccuracy)
        trainprescion.append(score.TrainPrecision)
        trainrecall.append(score.TrainRecall)
        trainf1.append(score.TrainF1)
        trainauc.append(score.TrainAUC)
     
    plotAccuracy(x, accuracy, trainaccuracy, title, xLabel, yLabel)
    plotF1(x, f1, trainf1, title, xLabel, yLabel)
    plotPrecision(x, prescion, trainprescion, title, xLabel, yLabel)
    plotRecall(x, recall, trainrecall, title, xLabel, yLabel) 

    plt.clf()
    plt.title = title
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 

    # plt.plot(x, accuracy, label='Accuracy', color='r', lw=1.0)   
    # # plt.plot(x, recall, label='Recall', color='g', lw=2.0)   
    # plt.plot(x, f1, label='F1', color='b', lw=1.0)  
    # # plt.plot(x, prescion, label='Precision', color='k', lw=2.0) 
    plt.plot(x, auc, label='AUC')  
    
    # plt.plot(x, trainaccuracy, label='Train Accuracy', color='r',  lw=5.0,  alpha=0.25)   
    # # plt.plot(x, trainprescion, label='Train Precision', color='g', marker='o', lw=1.0, ls='--')   
    # plt.plot(x, trainf1, label='Train F1', color='b',  lw=5.0,  alpha=0.25)  
    # # plt.plot(x, trainrecall, label='Train Recall', color='k', marker='o', lw=1.0, ls='--') 
    plt.plot(x, trainauc, label='Train AUC')  

    plt.legend()
    plt.grid() 
    # plt.ylim(0, 1)
    # plt.show()

    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + ' AUC' + '.png'
        plt.savefig(name)

def plotF1(x, yTrain, yTest, title, xLabel, yLabel):
    plt.clf()
    plt.title = title
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 
 
    plt.plot(x, yTrain, label='F1') 
    plt.plot(x, yTest, label='Train F1')  

    plt.legend()
    plt.grid()  

    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + ' F1 ' + '.png'
        plt.savefig(name)

def plotAccuracy(x, yTrain, yTest, title, xLabel, yLabel):
    plt.clf()
    plt.title = title
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 
 
    plt.plot(x, yTest, label='Accuracy')    
    plt.plot(x, yTrain, label='Train Accuracy')    

    plt.legend()
    plt.grid()  

    if showPlot:
        plt.show()
    else:
        name = baseGraphPath  + title + ' Accuracy '+ '.png'
        plt.savefig(name)

def plotPrecision(x, yTrain, yTest, title, xLabel, yLabel):
    plt.clf()
    plt.title = title
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 
 
    plt.plot(x, yTest, label='Precision') 
    plt.plot(x, yTrain, label='Train Precision')  

    plt.legend()
    plt.grid()  

    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + ' Precision ' + '.png'
        plt.savefig(name)

def plotRecall(x, yTrain, yTest, title, xLabel, yLabel):
    plt.clf()
    plt.title = title
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 
 
    plt.plot(x, yTest, label='Recall') 
    plt.plot(x, yTrain, label='Train Recall')  

    plt.legend()
    plt.grid()  

    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + ' Recall ' + '.png'
        plt.savefig(name)

def plotValidationCurve(model, xTrain, yTrain, param , param_range, graphTitle, xRange=None):
    # plt.clf()
    # plt.figure(figsize=(6.25, 5))

    graphTitle = graphTitle + ' Validation Curve'

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot() 
    ax1.set_title(graphTitle)

 

    train_scores, valid_scores = validation_curve(model, xTrain, yTrain, param,
                                                param_range,
                                                cv=5)



    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)

    # diff = np.diff(train_scores_mean, test_scores_mean)
    # bestScore = 
 
    plt.xlabel(param)
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    lw = 2
    if xRange:
        param_range = xRange

    ax1.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    ax1.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    ax1.plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    ax1.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")

    
    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + graphTitle + '.png'
        plt.savefig(name)

def plotLearningCurves(estimator, title, xTrain, yTrain, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    title = title + ' Learning Curve'
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    # if axes is None:
    #     axes = plt.subplots(1, 1, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, xTrain, yTrain, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    
    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + '.png'
        plt.savefig(name)

def plotLearningCurve(estimator, title, xTrain, yTrain, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
 
    # plt.clf()
    # plt.figure(figsize=(6.25, 5))
     
    title = title + ' Learning Curve' 
 
    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot() 
    ax1.set_title(title)
    plt.xlabel('Training examples')
    plt.ylabel('Score') 
 
    if ylim is not None:
        ax1.plot.set_ylim(*ylim)
 

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, xTrain, yTrain, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    ax1.grid()
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax1.legend(loc="best")
 
    
    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + '.png'
        plt.savefig(name)

def plotConfusion(model, dataTitle, classNames, xTest, yTest):
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [(dataTitle + " Confusion matrix", None)]
    titles_options = [(dataTitle + " Confusion matrix", None, ''),
                    (dataTitle + " Confusion matrix", 'true', '')]

    for title, normalize, extra in titles_options:
        disp = plot_confusion_matrix(model, xTest, yTest,
                                    display_labels=classNames,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        # plt.figure(figsize=(6.25, 5))
        if showPlot:
            plt.show()
        else:
            name = baseGraphPath + title + ' extra '  + '.png'
            plt.savefig(name)


def plotAll(model, title, param, param_range, xTrain, yTrain, xTest, yTest):
   
    xLabel = param
    scoreList = util.ScoreList(xLabel) 
    bestScores = []
    for i in param_range:
        params = {param:i}
        model.set_params(**params) 
        model.fit(xTrain, yTrain)
        trainPred = model.predict(xTrain)
        testPred = model.predict(xTest) 

        score = scoreList.Add(yTest, testPred, yTrain, trainPred, i)
        bestScores.append([i, score.Accuracy])
 
    plotScores(scoreList.GetScores(), title=title, xLabel= xLabel)
    bestScores = sorted(bestScores, key= lambda x: x[1], reverse=True)
    return bestScores[0]

def plot(x, y, xLabel, yLabel, title):

     
    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot() 
    ax1.set_title(title) 
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 
    
    ax1.plot(x, y)  
   
    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + '.png'
        plt.savefig(name)

def plotAUC(model, title, param, param_range, xTrain, yTrain, xTest, yTest):
  
    title = title + ' AUC'

    train_results = []
    test_results = []
    for i in param_range:
        params = {param:i}
        model.set_params(**params) 
        model.fit(xTrain, yTrain)
        yTrainPred = model.predict(xTrain)
        yTestPred = model.predict(xTest)
 
        false_positive_rate, true_positive_rate, thresholds = roc_curve(yTrain, yTrainPred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest, yTestPred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(param_range, train_results, 'b', label='Train AUC')
    line2, = plt.plot(param_range, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel(param)
    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + '.png'
        plt.savefig(name)
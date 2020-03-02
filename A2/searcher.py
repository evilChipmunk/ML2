
from sklearn import svm, datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import numpy as np

def basicResults(clfObj,trgX,trgY,tstX,tstY,params,clf_type=None,dataset=None):
    np.random.seed(55)
    if clf_type is None or dataset is None:
        raise
    cv = GridSearchCV(clfObj,n_jobs=1,param_grid=params,refit=True,verbose=10,cv=5,scoring=scorer)
    cv.fit(trgX,trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/{}_{}_reg.csv'.format(clf_type,dataset),index=False)
    test_score = cv.score(tstX,tstY)
    with open('./output/test results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(clf_type,dataset,test_score,cv.best_params_))    
    N = trgY.shape[0]    
    curve = ms.learning_curve(cv.best_estimator_,trgX,trgY,cv=5,train_sizes=[50,100]+[int(N*x/10) for x in range(1,8)],verbose=10,scoring=scorer)
    curve_train_scores = pd.DataFrame(index = curve[0],data = curve[1])
    curve_test_scores  = pd.DataFrame(index = curve[0],data = curve[2])
    curve_train_scores.to_csv('./output/{}_{}_LC_train.csv'.format(clf_type,dataset))
    curve_test_scores.to_csv('./output/{}_{}_LC_test.csv'.format(clf_type,dataset))
    return cv




def searchSVMLinear(xTrain, yTrain, xTest, yTest):
    parameters = {'kernel': ['linear']
                    , 'C': [.01, .1, 1, 10, 100, 1000]
                    , 'gamma': ['scale', 'auto', 0.01,0.05,0.1,1,10,50,100]  
                      }
    model = SVC(cache_size=5000, max_iter=5000, coef0=0, shrinking=True, decision_function_shape='ovo')
    return search(xTrain, yTrain, xTest, yTest, parameters, model)
    
def searchSVMPoly(xTrain, yTrain, xTest, yTest):
    parameters = {'kernel': ['poly']
                    , 'degree': [1, 2, 3, 4, 5, 10] 
                    , 'C': [0.01, 0.1, 1, 10, 100, 1000]
                    , 'gamma': ['scale', 'auto', 0.01,0.05,0.1,1,10,50,100]   
                     }
    model = SVC(cache_size=5000, max_iter=5000, coef0=0, shrinking=True, decision_function_shape='ovo')
    return search(xTrain, yTrain, xTest, yTest, parameters, model)
    
def searchSVMRBF(xTrain, yTrain, xTest, yTest):
    parameters = {'kernel': ['rbf']
                    , 'C': [.01, .1, 1, 10, 100, 1000]
                    , 'gamma': ['scale', 'auto', 0.01,0.05,0.1,1,10,50,100]    
                     }
    model = SVC(cache_size=5000, max_iter=5000, coef0=0, shrinking=True, decision_function_shape='ovo')
    return search(xTrain, yTrain, xTest, yTest, parameters, model)
    
def searchSVMSigmoid(xTrain, yTrain, xTest, yTest):
    parameters = {'kernel': ['sigmoid']
                    , 'C': [.01, .1, 1, 10, 100, 1000]
                    , 'gamma': ['scale', 'auto', 0.01,0.05,0.1,1,10,50,100]   
                      }
    model = SVC(cache_size=5000, max_iter=5000, coef0=0, shrinking=True, decision_function_shape='ovo')
    return search(xTrain, yTrain, xTest, yTest, parameters, model)




# def searchSVMLinear(xTrain, yTrain, xTest, yTest):
#     parameters = {'kernel': ['linear'],
#                     'C': [1, 10, 100, 1000], 
#                      'gamma': ['scale', 'auto'] , 'coef0': [0, 1, 2, 3, 4, 5, 10],
#                       'shrinking': [True,False] , 'decision_function_shape': ('ovo', 'ovr') }
#     return search(xTrain, yTrain, xTest, yTest, parameters, SVC(cache_size=5000, max_iter=5000))
    
# def searchSVMPoly(xTrain, yTrain, xTest, yTest):
#     parameters = {'kernel': ['poly'],
#                     'C': [1, 10, 100, 1000], 'degree': [1, 2, 3, 4, 5, 10],
#                      'gamma': ['scale', 'auto'] , 'coef0': [0, 1, 2, 3, 4, 5, 10],
#                       'shrinking': [True,False] , 'decision_function_shape': ('ovo', 'ovr') }
#     return search(xTrain, yTrain, xTest, yTest, parameters, SVC(cache_size=5000, max_iter=5000))
    
# def searchSVMRBF(xTrain, yTrain, xTest, yTest):
#     parameters = {'kernel': ['rbf'],
#                     'C': [1, 10, 100, 1000], 
#                      'gamma': ['scale', 'auto'] , 'coef0': [0, 1, 2, 3, 4, 5, 10],
#                       'shrinking': [True,False] , 'decision_function_shape': ('ovo', 'ovr') }
#     return search(xTrain, yTrain, xTest, yTest, parameters, SVC(cache_size=5000, max_iter=5000))
    
# def searchSVMSigmoid(xTrain, yTrain, xTest, yTest):
#     parameters = {'kernel': ['sigmoid'],
#                     'C': [1, 10, 100, 1000], 
#                      'gamma': ['scale', 'auto'] , 'coef0': [0, 1, 2, 3, 4, 5, 10],
#                       'shrinking': [True,False] , 'decision_function_shape': ('ovo', 'ovr') }
#     return search(xTrain, yTrain, xTest, yTest, parameters, SVC(cache_size=5000, max_iter=5000))






def searchKNN(xTrain, yTrain, xTest, yTest):
    parameters = {'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
    'p': [1, 2] 
    }
    return search(xTrain, yTrain, xTest, yTest, parameters, KNeighborsClassifier())

def searchNetwork(xTrain, yTrain, xTest, yTest):
    parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'] ,
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'] #,
   # 'momentum': [np.linspace(0, 1, .1)] 
    }

    # Invalid parameter criterion for estimator MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
    #           beta_2=0.999, early_stopping=False, epsilon=1e-08,
    #           hidden_layer_sizes=(18, 1), learning_rate='constant',
    #           learning_rate_init=0.001, max_iter=250, momentum=0.9,
    #           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
    #           random_state=None, shuffle=True, solver='adam', tol=0.0001,
    #           validation_fraction=0.1, verbose=False, warm_start=False).


    return search(xTrain, yTrain, xTest, yTest, parameters, MLPClassifier())

def searchDT(xTrain, yTrain, xTest, yTest):
    parameters = { 
                'criterion':['gini','entropy']
                # , 'max_depth': np.arange(1, 50)
                , 'max_features': ['auto', 'sqrt', 'log2', None]
                ,'splitter': ['best', 'random']
                }

    return search(xTrain, yTrain, xTest, yTest, parameters, DecisionTreeClassifier())

def searchAda(xTrain, yTrain, xTest, yTest):
  
    parameters = { 'algorithm':['SAMME','SAMME.R']}

    return search(xTrain, yTrain, xTest, yTest, parameters, AdaBoostClassifier())


def search(xTrain, yTrain, xTest, yTest, parameters, model): 
    clf = GridSearchCV( model, parameters)
    clf.fit(xTrain, yTrain)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    return clf.best_params_

def searchAndReport(xTrain, yTrain, xTest, yTest, parameters, model): 
 
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV( model, parameters, scoring='%s_macro' % score )
        clf.fit(xTrain, yTrain)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_) 
        print()
      
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
 
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = yTest, clf.predict(xTest)
        print(classification_report(y_true, y_pred))
        print()
 
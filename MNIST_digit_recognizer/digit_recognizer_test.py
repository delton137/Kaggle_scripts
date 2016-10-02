#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier  


def test_all_the_classifiers(x, y):
    ''' test a bunch of classifiers and print out a sorted list of CV accuracies
        inputs: 
            x: training data features, numpy array or Pandas dataframe
            y: training data labels, numpy array or Pandas dataframe
    '''
    #gammas = np.logspace(-6, -1, 10)
    #svc = SVC()
    #SVCgrid_search = grid_search.GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas), #n_jobs=-1)
    
    classifier_dict = {
                #'gridCV': SVCgrid_search,
                #'linear SVC': SVC(kernel="linear"),
                #'polynomial SVC': SVC(kernel="poly", C=0.025),
                'Multi layer perceptron' :  MLPClassifier(),
                'radial basis function SVC': SVC(gamma=2, C=1),
                'logistic regression':LogisticRegression(fit_intercept=False, penalty='l1'),
                'k nearest neighbors': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(max_depth=5),
                'Random Forest': RandomForestClassifier(max_depth=5, n_estimators=200),
                'AdaBoost': AdaBoostClassifier(),
                'Naive Bayes, Gaussian': GaussianNB(),
                'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
                }
    
    
    results = {}
    
    for (name, model) in classifier_dict.items():
        print("running %s" % name)
        model.fit(x, y)
        scores = cross_validation.cross_val_score(model, x, y, cv=3, n_jobs=-1)
        mean_score = scores.mean()
        results[name] = mean_score
        print("avg accuracy = %4.3f" % mean_score)
    
    sorted_names = sorted(results, key=results.__getitem__, reverse=True)
    sorted_scores = sorted(results.values(), reverse=True)

    for i in range(len(sorted_scores)):
        print("%20s %4.3f" % (sorted_names[i],sorted_scores[i]))
     
      
if __name__ == '__main__':
    train_df = pd.read_csv("train.csv")
    x_train = train_df.drop(['label'], axis=1).values
    y_train = train_df['label'].values
    
    test_all_the_classifiers(x_train, y_train)
    
             
            
    
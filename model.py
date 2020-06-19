from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint 
import pandas as pd
import nltk
import re
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB



# ---------------------- #
#        Modeling        #
# ---------------------- #

def run_decision_tree(X_train, y_train):
    '''
    Function to run a decision tree model. The function creates the model, then uses 
    cross-validation grid search to figure out what the best parameters are. Returns a grid (object
    used to find best hyperparameters), df_result (holds the accuracy score for all hyperparameter values)
    and model (holds the model with the best hyperparameters, used to create predictions)
    '''


    #  keys are names of hyperparams, values are a list of values to try for that hyper parameter
    params = {
        'max_depth': range(1, 11),
        'criterion': ['gini', 'entropy']
    }

    dtree = DecisionTreeClassifier()

    # cv=4 means 4-fold cross-validation, i.e. k = 4
    grid = GridSearchCV(dtree, params, cv=10)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    results = grid.cv_results_

    for score, p in zip(results['mean_test_score'], results['params']):
        p['score'] = score
    df_result = pd.DataFrame(results['params'])
    
    print(grid.best_params_)

    return grid, df_result, model

def run_random_forest(X_train, y_train):
    '''
    Function to run a random forest model. The function creates the model, then uses 
    cross-validation grid search to figure out what the best parameters are. Returns a grid (object
    used to find best hyperparameters), df_result (holds the accuracy score for all hyperparameter values)
    and model (holds the model with the best hyperparameters, used to create predictions)
    '''
    
    params = {
    'max_depth': range(1, 20),
    "min_samples_leaf": range(1,20)
    }

    rf = RandomForestClassifier(random_state = 123)

    # cv=4 means 4-fold cross-validation, i.e. k = 4
    grid = GridSearchCV(rf, params, cv=10)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    results = grid.cv_results_

    for score, p in zip(results['mean_test_score'], results['params']):
        p['score'] = score
    df_result = pd.DataFrame(results['params'])
    
    print(grid.best_params_)

    return grid, df_result, model

def run_knn(X_train, y_train):
    '''
    Function to run a knn model. The function creates the model, then uses 
    cross-validation grid search to figure out what the best parameters are. Returns a grid (object
    used to find best hyperparameters), df_result (holds the accuracy score for all hyperparameter values)
    and model (holds the model with the best hyperparameters, used to create predictions)
    '''

    knn = KNeighborsClassifier()

    params = {
        'weights': ["uniform", "distance"],
        "n_neighbors": range(1,20)
    }

    # cv=4 means 4-fold cross-validation, i.e. k = 4
    grid = GridSearchCV(knn, params, cv=10)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    results = grid.cv_results_

    for score, p in zip(results['mean_test_score'], results['params']):
        p['score'] = score
    df_result = pd.DataFrame(results['params'])
    
    print(grid.best_params_)

    return grid, df_result, model

def evaluate_on_test_data(X_test, y_test, model):
    model.score(X_test, y_test)

def create_prediction(df, model):
    y_pred = model.predict(df)
    return y_pred


##############
# Evaluation #
##############
def create_report(y_train, y_pred):
    '''
    Helper function used to create a classification evaluation report, and return it as df
    '''
    report = classification_report(y_train, y_pred, output_dict = True)
    report = pd.DataFrame.from_dict(report)
    return report


def accuracy_report(model, y_pred, y_train):
    '''
    Main function used to create printable versions of the classification accuracy score, confusion matrix and classification report.
    '''
    report = classification_report(y_train, y_pred, output_dict = True)
    report = pd.DataFrame.from_dict(report)
    accuracy_score = f'Accuracy on dataset: {report.accuracy[0]:.2f}'

    labels = sorted(y_train.unique())
    matrix = pd.DataFrame(confusion_matrix(y_train, y_pred), index = labels, columns = labels)

    return accuracy_score, matrix, report

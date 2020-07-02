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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans

import preprocessing_permits as pr
import numpy as np

# ---------------------- #
#        Modeling        #
# ---------------------- #

def run_decision_tree_cv(X_train, y_train):
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
    grid = GridSearchCV(dtree, params, cv=3, scoring= "recall")
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    results = grid.cv_results_

    for score, p in zip(results['mean_test_score'], results['params']):
        p['score'] = score
    df_result = pd.DataFrame(results['params'])
    
    print(grid.best_params_)

    return grid, df_result, model

def run_random_forest_cv(X_train, y_train):
    '''
    Function to run a random forest model. The function creates the model, then uses 
    cross-validation grid search to figure out what the best parameters are. Returns a grid (object
    used to find best hyperparameters), df_result (holds the accuracy score for all hyperparameter values)
    and model (holds the model with the best hyperparameters, used to create predictions)
    '''
    
    params = {
    'max_depth': range(1, 10),
    "min_samples_leaf": range(1,10)
    }

    rf = RandomForestClassifier(random_state = 123)

    # cv=4 means 4-fold cross-validation, i.e. k = 4
    grid = GridSearchCV(rf, params, cv=3, scoring= "recall")
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    results = grid.cv_results_

    for score, p in zip(results['mean_test_score'], results['params']):
        p['score'] = score
    df_result = pd.DataFrame(results['params'])
    
    print(grid.best_params_)

    return grid, df_result, model

def run_knn_cv(X_train, y_train):
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
    grid = GridSearchCV(knn, params, cv=3, scoring= "recall")
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


############################################################################################################

                                        # Traditional Modeling #

############################################################################################################

# ---------------------- #
#        Modeling        #
# ---------------------- #

# Decision Tree

def run_clf(X_train, y_train, max_depth):
    '''
    Function used to create and fit decision tree models. It requires a max_depth parameter. Returns model and predictions.
    '''
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=123)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    return clf, y_pred

def run_clf_loop(train_scaled, validate_scaled, y_validate, y_train, max_range):
    for i in range(1, max_range):
        clf, y_pred = run_clf(train_scaled, y_train, i)
        score = clf.score(train_scaled, y_train)
        validate_score = clf.score(validate_scaled, y_validate)
        _, _, report = accuracy_report(clf, y_pred, y_train)
        recall_score = report["True"].recall
        print(f"Max_depth = {i}, accuracy_score = {score:.2f}. validate_score = {validate_score:.2f}, recall = {recall_score:.2f}")

def clf_feature_importances(clf, train_scaled):
    coef = clf.feature_importances_
    # We want to check that the coef array has the same number of items as there are features in our X_train dataframe.
    assert(len(coef) == train_scaled.shape[1])
    coef = clf.feature_importances_
    columns = train_scaled.columns
    df = pd.DataFrame({"feature": columns,
                    "feature_importance": coef,
                    })

    df = df.sort_values(by="feature_importance", ascending=False)
    sns.barplot(data=df, x="feature_importance", y="feature", palette="Blues_d")
    plt.title("What are the most influencial features?")

# KNN

def run_knn(X_train, y_train, n_neighbors):
    '''
    Function used to create and fit KNN model. Requires to specify n_neighbors. Returns model and predictions.
    '''
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    return knn, y_pred

def run_knn_loop(train_scaled, validate_scaled, y_validate, y_train, max_range):
    for i in range(1, max_range):
        knn, y_pred = run_knn(train_scaled, y_train, i)
        score = knn.score(train_scaled, y_train)
        validate_score = knn.score(validate_scaled, y_validate)
        _, _, report = accuracy_report(knn, y_pred, y_train)
        recall_score = report["True"].recall
        print(f"Max_depth = {i}, accuracy_score = {score:.2f}. validate_score = {validate_score:.2f}, recall = {recall_score:.2f}")


# Random_forest

def run_rf(X_train, y_train, leaf, max_depth):
    ''' 
    Function used to create and fit random forest models. Requires to specif leaf and max_depth. Returns model and predictions.
    '''
    rf = RandomForestClassifier(random_state= 123, min_samples_leaf = leaf, max_depth = max_depth).fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    return rf, y_pred

def run_rf_loop(train_scaled, validate_scaled, y_validate, y_train, max_range):
    for i in range(1, max_range):
        rf, y_pred = run_rf(train_scaled, y_train, 1, i)
        score = rf.score(train_scaled, y_train)
        validate_score = rf.score(validate_scaled, y_validate)
        _, _, report = accuracy_report(rf, y_pred, y_train)
        recall_score = report["True"].recall
        print(f"Max_depth = {i}, accuracy_score = {score:.2f}. validate_score = {validate_score:.2f}, recall = {recall_score:.2f}")

def rf_feature_importance(rf, train_scaled):
    coef = rf.feature_importances_
    columns = train_scaled.columns
    df = pd.DataFrame({"feature": columns,
                    "feature_importance": coef,
                    })

    df = df.sort_values(by="feature_importance", ascending=False)
    sns.barplot(data=df, x="feature_importance", y="feature", palette="Blues_d")
    plt.title("What are the most influencial features?")

# Logistic Regression

def run_lg(X_train, y_train):
    '''
    Function used to create and fit logistic regression models. Returns model and predictions.
    '''
    logit = LogisticRegression().fit(X_train, y_train)
    y_pred = logit.predict(X_train)
    return logit, y_pred

# ------------------ #
#    Predictions     #
# ------------------ #

def prep_prediction_data():
    df = pr.permits_preprocessing_mother_function()
    df = df[df.year > 1997]
   # bring clusters
    df, kmeans, centroids, scaler, scaled_ei_threshold_value, X = pr.create_clusters(df)

    df[["avg_units_per_bldg", "ei"]] = scaler.inverse_transform(df[["avg_units_per_bldg", "ei"]])

    # # create object
    # scaler = PowerTransformer()
    # # fit object
    # scaler.fit(df[["avg_units_per_bldg"]])
    # # transform using object
    # df["avg_units_per_bldg_scaled"] = scaler.transform(df[["avg_units_per_bldg"]])

    # scaler = PowerTransformer()
    # scaler.fit(df[["ei"]])
    # # transform using object
    # df["ei_scaled"] = scaler.transform(df[["ei"]])



    # # define features for KMeans modeling
    # X = df[["avg_units_per_bldg_scaled", "ei_scaled"]]

    # # cluster using k of 6

    # # create object
    # kmeans = KMeans(n_clusters=6, random_state=123)
    # # fit object
    # kmeans.fit(X)
    # # predict using object
    # df["cluster"] = kmeans.predict(X)

    df = df.merge(centroids, how="left", left_on="cluster", right_on=centroids.index)

    return df, kmeans

# Helper function used to updated the scaled arrays and transform them into usable dataframes
def return_values_prediction(scaler, df):
    train_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns.values).set_index([df.index.values])
    return scaler, train_scaled

# Linear scaler
def min_max_scaler_prediction(df):
    scaler = MinMaxScaler().fit(df)
    scaler, df_scaled = return_values_prediction(scaler, df)
    return scaler, df_scaled

def create_predictions_df(df, kmeans, knn):

    ## Found a mistake - can't just create an average of an aver_units_per_bld. Need to bring in both variables, create the avg, and then calculate the new avg_units_per_bldg
    # predictions = df[(df.year == 2018) | (df.year == 2019)].groupby("city_state")[["avg_units_per_bldg_x", "ei_x", "total_high_density_bldgs", "total_high_density_value"]].mean()

    # predictions = df[(df.year == 2018) | (df.year == 2019)].groupby("city_state")[["total_high_density_units", "ei_x", "total_high_density_bldgs", "total_high_density_value"]].mean()
    # predictions["avg_units_per_bldg"] = predictions["total_high_density_units"] / predictions["total_high_density_bldgs"]

    # create 2018 & 2019 masked DataFrame
    predictions = df[(df.year == 2018) | (df.year == 2019)]
    # create DataFrame for total units over 2018-2019
    total_units = pd.DataFrame(predictions.groupby(["city", "state"]).total_high_density_units.sum())
    # create DataFrame for total buildings over 2018-2019
    total_bldgs = pd.DataFrame(predictions.groupby(["city", "state"]).total_high_density_bldgs.sum())
    # create DataFrame for total value over 2018-2019
    total_value = pd.DataFrame(predictions.groupby(["city", "state"]).total_high_density_value.sum())
    # merging total_units to predictions
    predictions = predictions.merge(total_units, how="left", on=["city", "state"], suffixes=("_og", "_1819"))
    # merging total_bldgs to predictions
    predictions = predictions.merge(total_bldgs, how="left", on=["city", "state"], suffixes=("_og", "_1819"))
    # merging total_values to predictions
    predictions = predictions.merge(total_value, how="left", on=["city", "state"], suffixes=("_og", "_1819"))

    # 2018-2019 total units and buildings needed to calculate the proper weighted average
    predictions = predictions.groupby("city_state")[["total_high_density_units_1819", "total_high_density_bldgs_1819"]].mean()

    # masking initial df variable for last two years 
    # grouping by city_state to get 130 unique observations
    # calc mean for ei, total buildings, and total valuation
    avgs = df[(df.year == 2018) | (df.year == 2019)].groupby("city_state")[["ei_x", "total_high_density_bldgs", "total_high_density_value"]].mean()
    
    # predictions["avg_units_per_bldg"] = predictions["total_high_density_units_1819"] / predictions["total_high_density_bldgs_1819"]
    # predictions.drop(columns="total_high_density_units_1819", inplace=True)

    # calc weighted average number of units per building over 2018-2019
    avgs["avg_units_per_bldg"] = predictions["total_high_density_units_1819"] / predictions["total_high_density_bldgs_1819"]

    # create object
    scaler = PowerTransformer()
    # fit object
    scaler.fit(avgs[["avg_units_per_bldg", "ei_x"]])
    # transform using object
    avgs[["avg_units_per_bldg", "ei_x"]] = scaler.transform(avgs[["avg_units_per_bldg", "ei_x"]])
    
    # define features for KMeans modeling
    X = avgs[["avg_units_per_bldg", "ei_x"]]

    avgs["cluster"] = kmeans.predict(X)

    avgs[["avg_units_per_bldg", "ei_x"]] = scaler.inverse_transform(avgs[["avg_units_per_bldg", "ei_x"]])
    
    scaler, avgs_scaled = min_max_scaler_prediction(avgs)

    avgs["label"] = knn.predict(avgs_scaled)

    city = avgs.reset_index().city_state.str.split("_", n=1, expand=True)[0]

    state = avgs.reset_index().city_state.str.split("_", n=1, expand=True)[1]

    avgs = avgs.reset_index()

    avgs["city"] = city

    avgs["state"] = state


    df_best = (
        avgs[(avgs.label) & ((avgs.cluster == 0) | (avgs.cluster == 4))]
    )

    df_high_density = (
        avgs[(avgs.label) & ((avgs.cluster == 5) | (avgs.cluster == 2))]
    )

    df_stable_high_markets = (
        avgs[(avgs.label) & ((avgs.cluster == 3) | (avgs.cluster == 1))]
    )

    df_best["recommendation_label"] = "Best_ROI"

    df_high_density["recommendation_label"] = "medium_ROI"

    df_stable_high_markets["recommendation_label"] = "Stable_High"

    avgs["recommendation_label"] = np.nan

    avgs.recommendation_label = avgs.recommendation_label.fillna(df_best.recommendation_label)

    avgs.recommendation_label = avgs.recommendation_label.fillna(df_high_density.recommendation_label)

    avgs.recommendation_label = avgs.recommendation_label.fillna(df_stable_high_markets.recommendation_label)

    avgs.recommendation_label = avgs.recommendation_label.fillna("Not Recommended to Enter")

    return avgs
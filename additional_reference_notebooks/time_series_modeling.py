# data manipulation 
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import sklearn

# data visualization 
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt

from statsmodels.tsa.api import Holt

# FB Prophet
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.plot import plot_forecast_component

import model


def create_predictions_df(train, validate, target_variable):

    baseline = round(train[target_variable][-1:][0], 2)

    predictions = pd.DataFrame({
    "actual": validate[target_variable], 
    "baseline": [baseline]},
    index = validate.index)

    return predictions

# ------------ #
#   Modeling   #
# ------------ #

def run_simple_average(train, target_variable):
    y_pred = round(train[target_variable].mean(), 2)
    return y_pred 

def run_moving_average(train, target_variable, rolling_average):
    y_pred = round(train[target_variable].rolling(rolling_average).mean().iloc[-1], 2)
    return y_pred


def run_holts(train, validate, target_variable,exponential,  smoothing_level = .1, smoothing_slope = .1):
    # Create model object
    model = Holt(train[target_variable], exponential = exponential)

    # Fit model 
    model = model.fit(smoothing_level = smoothing_level, smoothing_slope=smoothing_slope, optimized = False)

    # Create predictions
    y_pred = model.predict(start=validate.index[0], end=validate.index[-1])

    return model, y_pred

def prep_fb(df, target_variable):
    
    df.index = df.index.rename("ds")
    df = df[[target_variable]]
    df.rename(columns = {target_variable: "y"}, inplace=True)
    return df

def run_prophet(train, validate, test, target_variable, cap, floor, initial, period, horizon):

    train["cap"] = cap
    train["floor"] = floor

    train = train.reset_index()
    validate = validate.reset_index()
    test = test.reset_index()

    m = Prophet(growth = 'logistic', 
            weekly_seasonality = True, 
            daily_seasonality = False,
            changepoint_range = 0.8)
    m.add_country_holidays(country_name='US')
    m.fit(train)

    df_cv = cross_validation(m, initial = initial, period = period, horizon = horizon)

    return m, df_cv

# __ Main Modeling Function __
def run_all_models(train, validate, target_variable, rolling_period, exponential=False):
    # Create predictions df and establish baseline
    predictions = create_predictions_df(train, validate, target_variable)

    # simple_average
    y_pred = run_simple_average(train, target_variable)
    predictions["simple_average"] = y_pred

    # moving average
    for i in rolling_period:
        y_pred = run_moving_average(train, target_variable, i)
        predictions["moving_average" + "_" + str(i)] = y_pred

    # holts prediction

    holt, y_pred = run_holts(train, validate, target_variable, exponential)
    predictions["holts_prediction"] = y_pred
    
    return predictions

def create_predictions_loop(target_variable, unique_city_states):
    # Going to need to create a loop, and create an average rmse score to see which one would be best

    results = pd.DataFrame()
    updates = 0
    for i in unique_city_states:
        # read data and create dataframe
        df, kmeans = model.prep_prediction_data()
        df["city_state"] = df.city + "_" + df.state
        modeling_df = df[df.city_state == i]
        
        # Create new dataframes
        modeling_df = modeling_df[["year", "city_state", target_variable]]
        modeling_df.year = pd.to_datetime(modeling_df.year, format = '%Y')
        modeling_df = modeling_df.set_index("year")
        # Split our data
        train = modeling_df[:"2015"]
        validate = modeling_df["2016":"2017"]
        test = modeling_df["2018":]
        # Model
        predictions = run_all_models(train, validate, target_variable, rolling_period = [1, 2, 3])
        # Evaluate
        rmse = predictions.apply(lambda col: sqrt(sklearn.metrics.mean_squared_error(predictions.actual, col)))
        rmse = pd.DataFrame(rmse, columns=[i])
        
        results = pd.concat([rmse, results], axis = 1)
        
    return results

# ---------------- #
#     Evaluate     #
# ---------------- # 

def print_rmse(model, predictions):
    print(f'RMSE = {round(sqrt(mean_squared_error(predictions.actual, predictions[model])), 0)}')

def plot_prediction(model, target_variable, train, validate, predictions):
    plt.figure(figsize = (20, 9))

    sns.lineplot(data=train, x=train.index, y=target_variable)
    sns.lineplot(data=validate, x=validate.index, y=target_variable)
    sns.lineplot(data=predictions, x=predictions.index, y=model)

def plot_rmse(predictions):
    rmse = predictions.apply(lambda col: sqrt(sklearn.metrics.mean_squared_error(predictions.actual, col)))
    rmse.plot.bar()

def print_rmse_plot(df_cv):
    fig = plot_cross_validation_metric(df_cv, metric = 'rmse', rolling_window = .1)


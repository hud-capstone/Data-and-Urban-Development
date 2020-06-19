import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import wrangle as wr

def get_model_df():
    """
    This function does the following:
    1. Calls the wrangle_hud function into the df variable
    2. Groups the data in df by the appropriate categorical variables
    3. Aggregates the grouped data using various metrics
    4. Sorts the data by sum and count of final_mortgage_amount
    5. Renames the features inplace
    6. Creates a label feature for classification
    """

    # calling wrangle_hud
    df = wr.wrangle_hud()

    # grouping by project_city, project_state, and fiscal_year_of_firm_commitment
    # aggregating final_mortgage_amount
    # sorting by sum and count in descending order
    model_df = df.groupby(["project_city", "project_state", "fiscal_year_of_firm_commitment_activity"])[
        "final_mortgage_amount"
    ].agg(["count", "sum", "mean", "median"]).sort_values(
        by=["sum", "count"], ascending=False
    ).reset_index()

    # rename the features
    model_df.rename(
        columns={
            "project_city": "city",
            "project_state": "state",
            "fiscal_year_of_firm_commitment_activity": "year",
            "count": "quantity_of_mortgages",
            "sum": "total_mortgage_amount",
            "mean": "average_mortgage_amount",
            "median": "median_mortgage_amount",
        },
        inplace=True,
    )

    # create label feature
    model_df["label"] = np.where(
        ((model_df.city == "Houston") & (model_df.state == "TX") & (model_df.year == 2009))
        | ((model_df.city == "Seattle") & (model_df.state == "WA") & (model_df.year == 2010))
        | ((model_df.city == "Dallas") & (model_df.state == "TX")  & (model_df.year == 2012)),
        True,
        False,
    )

    return model_df


# ------------------- #
# Feature Engineering #
# ------------------- #

def calculate_mortgage_var(df):
    df["mean_var"] = df.sort_values(["city", "state", "year"])["average_mortgage_amount"].pct_change() * 100
    return df

def calculate_mortgage_cnt_var(df):
    df["quantity_var"] = df.sort_values(["city", "state", "year"])["quantity_of_mortgages"].pct_change() * 100
    return df

def calculate_evolution_index(df):
    # EI = (1 + Company Growth %) / (1 + Market Growth %) X 100
    
    df["market_growth"] = df.groupby("year").mean_var.transform("mean")

    df["ei"] = (1+ df.mean_var) / (1+ df.market_growth) * 100

    return df

#__main prep__

def add_new_features(df):
    df = calculate_mortgage_var(df)
    df = calculate_mortgage_cnt_var(df)
    df = calculate_evolution_index(df)

    return df

# ------------------- #
#  Prep for Modeling  #
# ------------------- #

def train_test_data(df):
    train, test = train_test_split(df, train_size=.75, random_state=123)
    return train, test

#__Main Pre-modeling function__#
def prep_data_for_modeling(df, features_for_modeling):

    # To avoid Nan's, I have removed all data from 2006 (because all the var's would be nan)
    df_model = df[df.year > 2006]

    # Create an observation id to reduce the chance of mistake's
    df_model["observation_id"] = df_model.city + "_" + df_model.state + "_"  + df_model.year.astype(str)

    # select that features that we want to model, and use our observation id as the row id
    data = df_model[features_for_modeling].set_index("observation_id")

    train, test = train_test_data(data)

    X_train = train.drop(columns="label")
    y_train = train["label"]
    X_test = test.drop(columns="label")
    y_test = test["label"]

    return X_train, y_train, X_test, y_test


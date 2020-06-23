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

    # creating New Construction DataFrame
    nc_df = df[df.activity_description == "New Construction"]

    # grouping all data by project_city, project_state, and fiscal_year_of_firm_commitment
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
            "sum": "total_mortgage_volume",
            "mean": "average_mortgage_volume",
            "median": "median_mortgage_amount",
        },
        inplace=True,
    )

    # grouping nc_df in the same way as the model_df
    nc_model_df = (
        nc_df.groupby(
            ["project_city", "project_state", "fiscal_year_of_firm_commitment_activity"]
        )["final_mortgage_amount"]
        .agg(["count", "sum", "mean", "median"])
        .reset_index()
    )

    # rename the columns in the nc_model_df
    nc_model_df.rename(
        columns={
            "project_city": "city",
            "project_state": "state",
            "fiscal_year_of_firm_commitment_activity": "year",
            "count": "quantity_of_mortgages",
            "sum": "total_mortgage_volume",
            "mean": "average_mortgage_volume",
            "median": "median_mortgage_amount",
        },
        inplace=True,
    )

    # merging model_df and nc_model_df on city, state, and year and suffixing the features with the same names appropriately
    combined_df = pd.merge(model_df, nc_model_df, on=["city", "state", "year"], how="left", suffixes=("_pop", "_nc"))

    # reassigning the model_df variable to the combined DataFrame created in the cell above
    model_df = combined_df

    # filling 

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

def calculate_city_state_vol_delta(df):
    """
    This function creates the specific market growth (city + state observation) rate using volume by doing the following:
    1. Creates the city_state_growth_pop feature out of the total_mortgage_volume_pop
    2. Creates the city_state_growth_nc feature out of the total_mortgage_volume_nc
    3. Returns the df with the new features
    """

    # create city_state_vol_delta_pop
    df["city_state_vol_delta_pop"] = df.sort_values(["year"]).groupby(["city", "state"])[["total_mortgage_volume_pop"]].pct_change()

    # create city_state_vol_delta_nc
    df["city_state_vol_delta_nc"] = df.sort_values(["year"]).groupby(["city", "state"])[["total_mortgage_volume_nc"]].pct_change()

    return df

def calculate_city_state_qty_delta(df):
    """
    This function creates the specific market growth (city + state observation) rate using quantity by doing the following:
    1. Creates the city_state_qty_delta_pop feature out of the quantity_of_mortgages_pop
    2. Creates the city_state_qty_delta_nc feature out of the quantity_of_mortgages_nc
    3. Returns the df with the new features
    """

    # create city_state_qty_delta_pop
    df["city_state_qty_delta_pop"] = df.sort_values(["year"]).groupby(["city", "state"])[["quantity_of_mortgages_pop"]].pct_change()

    # create city_state_qty_delta_nc
    df["city_state_qty_delta_nc"] = df.sort_values(["year"]).groupby(["city", "state"])[["quantity_of_mortgages_nc"]].pct_change()

    return df

def calculate_evolution_index(df):
    """
    This function calculates the evolution index using the market_volume_delta feature created within using the market_volume
    """

    # EI = (1 + Company Growth %) / (1 + Market Growth %) X 100
    df = df.sort_values(["city", "state", "year"])

    # calc market_volume_pop for the year
    df["market_volume"] = df.groupby("year").total_mortgage_volume_pop.transform("sum")

    # calculate market growth rate for the population from prior year
    df["market_volume_delta"] = np.where(df.year > 2006, df["market_volume"].pct_change(), np.nan)

    # calc market_volume_nc for the year
    # df["market_volume_nc"] = df.groupby("year").total_mortgage_volume_nc.transform("sum")

    # calculate market growth rate from prior year
    # df["market_volume_delta_nc"] = np.where(df.year > 2006, df["market_volume_nc"].pct_change(), np.nan)

    # calc evolution index for population
    df["ei"] = (1 + df.city_state_vol_delta_pop) / (1 + df.market_volume_delta)

    # calc evolution index for new construction
    # df["ei_nc"] = (1 + df.city_state_vol_delta_nc) / (1 + df.market_volume_delta_nc)

    return df

#__main prep__

def add_new_features(df):
    df = calculate_city_state_vol_delta(df)
    df = calculate_city_state_qty_delta(df)
    df = calculate_evolution_index(df)

    return df

# ------------------- #
#  Prep for Modeling  #
# ------------------- #

def train_test_data(df):
    train, test = train_test_split(df, train_size=.75, random_state=123)
    return train, test

#__Main Pre-modeling function__#
def prep_data_for_modeling(df, features_for_modeling, label_feature):

    # To avoid Nan's, I have removed all data from 2006 (because all the var's would be nan)
    df_model = df[df.year > 2006]

    # Create an observation id to reduce the chance of mistake's
    df_model["observation_id"] = df_model.city + "_" + df_model.state + "_"  + df_model.year.astype(str)

    # select that features that we want to model, and use our observation id as the row id
    features_for_modeling += ["observation_id"]

    features_for_modeling += [label_feature]

    data = df_model[features_for_modeling].set_index("observation_id")

    train, test = train_test_data(data)
    train = train.sort_values("observation_id")
    test = test.sort_values("observation_id")

    X_train = train.drop(columns=label_feature)
    y_train = train[label_feature]
    X_test = test.drop(columns=label_feature)
    y_test = test[label_feature]

    return X_train, y_train, X_test, y_test

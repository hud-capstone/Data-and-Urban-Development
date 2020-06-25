import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

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

def filter_top_cities(df):

    df["city_state"] = df["city"] + "_" + df["state"]

    city_mask = df.groupby("city_state").year.count()

    city_mask = city_mask[city_mask == 15]

    # apply city mask to shrink the df
    def in_city_mask(x):
        return x in city_mask
    df = df[df.city_state.apply(in_city_mask)]

    df = df.sort_values(["city", "state", "year"])
    
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

# def train_test_data(df):
#     train, test = train_test_split(df, train_size=.75, random_state=123)
#     return train, test

# #__Main Pre-modeling function__#
# def prep_data_for_modeling(df, features_for_modeling, label_feature):

#     # To avoid Nan's, I have removed all data from 2006 (because all the var's would be nan)
#     df_model = df[df.year > 2006]

#     # Create an observation id to reduce the chance of mistake's
#     df_model["observation_id"] = df_model.city + "_" + df_model.state + "_"  + df_model.year.astype(str)

#     # select that features that we want to model, and use our observation id as the row id
#     features_for_modeling += ["observation_id"]

#     features_for_modeling += [label_feature]

#     data = df_model[features_for_modeling].set_index("observation_id")

#     train, test = train_test_data(data)
#     train = train.sort_values("observation_id")
#     test = test.sort_values("observation_id")

#     X_train = train.drop(columns=label_feature)
#     y_train = train[label_feature]
#     X_test = test.drop(columns=label_feature)
#     y_test = test[label_feature]

#     return X_train, y_train, X_test, y_test

def split_data(df, train_size=.75,random_state = 124):
    train, test = train_test_split(df, train_size=train_size, random_state=random_state, stratify = df["should_enter"])
    train, validate = train_test_split(train, train_size=train_size, random_state=random_state, stratify = train["should_enter"])
    return train, validate, test

def return_values(scaler, train, validate, test):
    '''
    Helper function used to updated the scaled arrays and transform them into usable dataframes
    '''
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

# Linear scaler
def min_max_scaler(train,validate, test):
    '''
    Helper function that scales that data. Returns scaler, as well as the scaled dataframes
    '''
    scaler = MinMaxScaler().fit(train)
    scaler, train_scaled, validate_scaled, test_scaled = return_values(scaler, train, validate, test)
    return scaler, train_scaled, validate_scaled, test_scaled

#__Main Pre-modeling function__#
def prep_data_for_modeling(df, features_for_modeling, label_feature):

    # To avoid Nan's, I have removed all data from 2006 (because all the var's would be nan)
    df_model = df[df.year > 2007]

    # Create an observation id to reduce the chance of mistake's
    df_model["observation_id"] = df_model.city + "_" + df_model.state + "_"  + df_model.year.astype(str)

    # select that features that we want to model, and use our observation id as the row id
    features_for_modeling += ["observation_id"]

    features_for_modeling += [label_feature]

    data = df_model[features_for_modeling].set_index("observation_id")
    
    train, validate, test = split_data(data)
    train = train.sort_values("observation_id")
    validate = validate.sort_values("observation_id")
    test = test.sort_values("observation_id")
    
    
    X_train = train.drop(columns=label_feature)
    y_train = train[label_feature]
    X_validate = validate.drop(columns=label_feature)
    y_validate = validate[label_feature]
    X_test = test.drop(columns=label_feature)
    y_test = test[label_feature]
    
    scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(X_train, X_validate, X_test)

    return train_scaled, validate_scaled, test_scaled, y_train, y_validate, y_test



#__Feature creation for labeling __#
def labeling_future_data(df):
    """this function takes in a data frame and returns a boolean column that identifies
    if a city_state_year is a market that should be entered"""
    
    df["label_quantity_of_mortgages_pop_2y"] = (df.sort_values(["year"])
                                  .groupby(["city", "state"])[["quantity_of_mortgages_pop"]]
                                  .pct_change(2)
                                  .shift(-2))
    
    df["label_total_mortgage_volume_pop_2y"] = (df.sort_values(["year"])
                                  .groupby(["city", "state"])[["total_mortgage_volume_pop"]]
                                  .pct_change(2)
                                  .shift(-2))
    
    Q3 = df.label_quantity_of_mortgages_pop_2y.quantile(.75)
    Q1 = df.label_quantity_of_mortgages_pop_2y.quantile(.25)
    upper_fence_quantity = Q3 + ((Q3-Q1)*1.5)
    upper_fence_quantity
    
    Q3 = df.label_total_mortgage_volume_pop_2y.quantile(.75)
    Q1 = df.label_total_mortgage_volume_pop_2y.quantile(.25)
    upper_fence_volume = Q3 + ((Q3-Q1)*1.5)
    upper_fence_volume
    
    df['should_enter'] = (df.label_total_mortgage_volume_pop_2y > upper_fence_volume) | (df.label_quantity_of_mortgages_pop_2y > upper_fence_quantity)
    
    return df

# Mother Preprocessing Function #

def preprocessing_main_function(features_for_modeling, label_feature):
    df = get_model_df()
    df = add_new_features(df)
    df = filter_top_cities(df)
    df = labeling_future_data(df)

    # Add oversampling
    df = df.append(df[df.should_enter])
    df = df.append(df[df.should_enter])
    df = df.append(df[df.should_enter])

    train_scaled, validate_scaled, test_scaled, y_train, y_validate, y_test = prep_data_for_modeling(df, features_for_modeling, label_feature)

    return train_scaled, validate_scaled, test_scaled, y_train, y_validate, y_test

# --------------- #
# Builing Permits #
# --------------- #

def prep_building_permits(df):
    
    df["city"] = df.cbsa_name.str.split("  ", 1, expand = True)[0]
    
    df["state"] = df.cbsa_name.str.split("  ", 1, expand = True)[1]
    
    df["major_city"] = df.city.str.split("-", 1, expand=True)[0]
    
    df["major_state"] = df.state.str.split("-", 1, expand=True)[0]
    
    df["metropolitan_area"] = df.state.str.split("-", 1, expand=True)[1]
    
    df["metropolitan_area"] = df.major_state.str.split(" ", 1, expand=True)[1]
    
    df["major_state"] = df.major_state.str.split(" ", 1, expand=True)[0]
    
    df = df.groupby(["major_city","major_state", "survey_date"]).sum().reset_index()
    
    return df

def get_permits_model_df():
    """
    Docstring
    """

    # calling wrangle_hud
    df = wr.acquire_building_permits()

    # call prep_building_permits
    df = prep_building_permits(df)

    # rename columns inplace
    df.rename(
        columns={
            "major_city": "city",
            "major_state": "state",
            "survey_date": "year",
            "five_or_more_units_bldgs_est": "total_high_density_bldgs",
            "five_or_more_units_units_est": "total_high_density_units",
            "five_or_more_units_value_est": "total_high_density_value",
        },
        inplace=True,
    )

    # select only high denisty estimation features
    df = df[
        [
            "city",
            "state",
            "year",
            "total_high_density_bldgs",
            "total_high_density_units",
            "total_high_density_value",
        ]
    ]

    # multiplying valuation metric by 1000 as the documentation states "valuation is shown in thousands of dollars"
    # helps alec's brain not hurt as bad
    df["total_high_density_value"] = df.total_high_density_value * 1000

    # sort values inplace
    df.sort_values(by=["city", "state", "year"], inplace=True)

    return df


# ---------------------------------------- #
# Feature Engineering for Building Permits #
# ---------------------------------------- #

def calculate_avg_units_per_bldg(df):
    """
    Docstring
    """

    # divide the total number of units per city + state + year by the total number of buildings per city + state + year
    df["avg_units_per_bldg"] = df["total_high_density_units"] / df["total_high_density_bldgs"]

    return df

def calculate_value_per_bldg(df):
    """
    Docstring
    """

    # divide the total valuation per city + state + year by the total number of buildings per city + state + year
    df["value_per_bldg"] = df["total_high_density_value"] / df["total_high_density_bldgs"]

    return df

def calculate_value_per_unit(df):
    """
    Docstring
    """

    # divide the total valuation per city + state + year by the total number of units per city + state + year
    df["value_per_unit"] = df["total_high_density_value"] / df["total_high_density_units"]

    return df

def calculate_city_state_high_density_bldgs_delta_pct(df):
    """
    Docstring
    """

    # create city_state_high_density_bldgs_delta_pct feature using the estimated high density (5 or more) structures
    df["city_state_high_density_bldgs_delta_pct"] = df.sort_values(["year"]).groupby(["city", "state"])[["total_high_density_bldgs"]].pct_change()

    return df

def calculate_city_state_high_density_units_delta_pct(df):
    """
    Docstring
    """

    # create city_state_high_density_units_delta_pct feature using the estimated high density (5 or more) units
    df["city_state_high_density_units_delta_pct"] = df.sort_values(["year"]).groupby(["city", "state"])[["total_high_density_units"]].pct_change()

    return df

def calculate_city_state_high_density_value_delta_pct(df):
    """
    Docstring
    """

    # create city_state_high_density_value_delta_pct feature using the estimated valuation of high density (5 or more) structures
    df["city_state_high_density_value_delta_pct"] = df.sort_values(["year"]).groupby(["city", "state"])[["total_high_density_value"]].pct_change()

    return df

def calculate_evolution_index(df):
    """
    This function calculates the evolution index using the market_volume_delta feature created within using the market_volume
    """

    # EI = (1 + Company Growth %) / (1 + Market Growth %) X 100
    df = df.sort_values(["city", "state", "year"])

    # calc market_volume for the year using the estimated valuation of high density (5 or more) structures
    df["market_volume"] = df.groupby("year").total_high_density_value.transform("sum")

    # calculate market growth rate for the population from prior year
    df["market_volume_delta_pct"] = np.where(df.year > 1980, df["market_volume"].pct_change(), np.nan)

    # calc evolution index for population
    df["ei"] = (1 + df.city_state_high_density_value_delta_pct) / (1 + df.market_volume_delta_pct)

    return df

#__main prep__#

def add_new_features(df):
    """
    Docstring
    """

    # call feature engineering functions
    df = calculate_avg_units_per_bldg(df)
    df = calculate_value_per_bldg(df)
    df = calculate_value_per_unit(df)
    df = calculate_city_state_high_density_bldgs_delta_pct(df)
    df = calculate_city_state_high_density_units_delta_pct(df)
    df = calculate_city_state_high_density_value_delta_pct(df)
    df = calculate_evolution_index(df)

    return df

def filter_top_cities_building_permits(df):
    """
    This function masks df in two ways:
    city_mask returns cities with only continuously reported data
    threshold_mask returns cities where they had at least one "5 or more unit" building permit for every year
    Returns 130 cities for modeling
    """
    df["city_state"] = df["city"] + "_" + df["state"]
    
    city_mask = df.groupby("city_state").year.count()
    
    city_mask = city_mask[city_mask == 23]
    
    # apply city mask to shrink the df
    def in_city_mask(x):
        return x in city_mask
    
    df = df[df.city_state.apply(in_city_mask)]
    
    threshold_mask = df.groupby('city_state').total_high_density_bldgs.agg(lambda x: (x == 0).sum())
    
    threshold_mask = threshold_mask[threshold_mask < 1].index.tolist()
    
    # apply threshold mask to shrink the df 
    def in_threshold_mask(x):
        return x in threshold_mask
    
    df = df[df.city_state.apply(in_threshold_mask)]
    
    df = df.sort_values(["city", "state", "year"])

    # reset index inplace
    df.reset_index(inplace=True)

    # drop former index inplace
    df.drop(columns=["index"], inplace=True)
    
    return df

def labeling_future_data(df):
    """this function takes in a data frame and returns a boolean column that identifies
    if a city_state_year is a market that should be entered"""
    
    df["five_or_more_units_bldgs_est_2y"] = (df.sort_values(["year"])
                                  .groupby(["city", "state"])[["total_high_density_bldgs"]]
                                  .pct_change(2)
                                  .shift(-2))
    
    df["five_or_more_units_value_est_2y"] = (df.sort_values(["year"])
                                  .groupby(["city", "state"])[["total_high_density_value"]]
                                  .pct_change(2)
                                  .shift(-2))
    
    Q3 = df.five_or_more_units_bldgs_est_2y.quantile(.75)
    
    Q1 = df.five_or_more_units_bldgs_est_2y.quantile(.25)
    
    upper_fence_quantity = Q3 + ((Q3-Q1)*1.5)
    
    Q3 = df.five_or_more_units_value_est_2y.quantile(.75)
    
    Q1 = df.five_or_more_units_value_est_2y.quantile(.25)
    
    upper_fence_volume = Q3 + ((Q3-Q1)*1.5)
    
    df['should_enter'] = (df.five_or_more_units_value_est_2y > upper_fence_volume) | (df.five_or_more_units_bldgs_est_2y > upper_fence_quantity)
    
    return df

def permits_preprocessing_mother_function():
    """
    Docstring
    """

    # call get_permits_model_df
    df = get_permits_model_df()

    # feature engineering
    df = add_new_features(df)

    # filter top cities
    df = filter_top_cities_building_permits(df)

    # label data
    df = labeling_future_data(df)

    return df
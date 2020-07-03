import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import wrangle as wr



# --------------- #
# Builing Permits #
# --------------- #

def get_permits_model_df():
    """
    Prep function used to actually read the data, apply column name changes, and sort values.
    """

    # calling wrangle_hud
    df = wr.acquire_building_permits()

    # call prep_building_permits
    df = wr.prep_building_permits(df)

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
    df["market_volume_delta_pct"] = np.where(df.year > 1997, df["market_volume"].pct_change(), np.nan)

    # calc evolution index for population
    df["ei"] = (1 + df.city_state_high_density_value_delta_pct) / (1 + df.market_volume_delta_pct)

    return df

# ---------------- #
#    Clustering    #
# ---------------- #


def create_clusters(df):
    """
    This function creates clusters using average units per building & evolution index.

    Important to note the the ei and average units per building are returned scaled. This means that if these features will be used for modeling, or exploration purposes other than with clusters, the values need to be inversed scaled
    """

    # mask df to exclude 1997 (no prior year growth measures)
    df = df[df.year > 1997]

    # this line of code gets the index from the first observation where the unscaled ei is greater than or equal to 1 and stores it in the
    # unscaled_ei_threshold_index variable
    unscaled_ei_threshold_index = df[df.ei >= 1].sort_values(by=["ei"]).head(1).index[0]

    # scale the features

    # create object
    scaler = PowerTransformer()
    # fit object
    scaler.fit(df[["avg_units_per_bldg", "ei"]])
    # transform using object
    df[["avg_units_per_bldg", "ei"]] = scaler.transform(df[["avg_units_per_bldg", "ei"]])

    # define features for KMeans modeling
    X = df[["avg_units_per_bldg", "ei"]]

    # cluster using k of 6

    # create object
    kmeans = KMeans(n_clusters=6, random_state=123)
    # fit object
    kmeans.fit(X)
    # predict using object
    df["cluster"] = kmeans.predict(X)

    # create centriods object
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    # this line gets the value of the scaled ei where the index of the scaled df is equal to the unscaled_ei_threshold_index
    scaled_ei_threshold_value = df[df.index == unscaled_ei_threshold_index]["ei"].values[0]

    return df, kmeans, centroids, scaler, scaled_ei_threshold_value, X

# ---------------- #
#     Data Prep    #
# ---------------- #

def add_new_features(df):
    """
    Prep function that calls all the individual feature engineering funtions and applies them to the dataframe. 
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

def labeling_future_data(df): ## Function used for the MVP ##
    """this function takes in a data frame and returns a boolean column that identifies
    if a city_state_year is a market that should be entered"""
    
    df.sort_values(by=['city', 'state', 'year'])
    
    df["bldgs_two_yr_growth_rate"] = (df.sort_values(["year"])
                                  .groupby(["city", "state"])[["total_high_density_bldgs"]]
                                  .pct_change(2))
    
    df["value_two_yr_growth_rate"] = (df.sort_values(["year"])
                                  .groupby(["city", "state"])[["total_high_density_value"]]
                                  .pct_change(2))

    df['future_bldgs_two_yr_growth_rate'] = df["bldgs_two_yr_growth_rate"].shift(-2)
                                  
    df['future_value_two_yr_growth_rate'] = df["value_two_yr_growth_rate"].shift(-2)
    
    Q3 = df.future_bldgs_two_yr_growth_rate.quantile(.75)
    
    Q1 = df.future_bldgs_two_yr_growth_rate.quantile(.25)
    
    upper_fence_quantity = Q3 + ((Q3-Q1)*1.5)
    
    Q3 = df.future_value_two_yr_growth_rate.quantile(.75)
    
    Q1 = df.future_value_two_yr_growth_rate.quantile(.25)
    
    upper_fence_volume = Q3 + ((Q3-Q1)*1.5)
    
    df['should_enter'] = (df.future_value_two_yr_growth_rate > upper_fence_volume) | (df.future_bldgs_two_yr_growth_rate > upper_fence_quantity)
    
    return df

# ------------- #
#   Splitting   #
# ------------- #

def split_data(df, train_size=.75,random_state = 124):
    train, test = train_test_split(df, train_size=train_size, random_state=random_state, stratify = df["should_enter"])
    train, validate = train_test_split(train, train_size=train_size, random_state=random_state, stratify = train["should_enter"])
    return train, validate, test

# ------------- #
#    Scaling    #
# ------------- #

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

# ----------------- #
#    Add labels     #
# ----------------- #

def create_cluster_labels(df):
    '''
    Function used to add labels to the data. The function takes in a dataframe, calculates the clusters, and then applies the logic to add the labels to the data. 

    The dataframe is return with the ie and the avg_units_per_bldg unscaled, so that it can be used for modeling.

    '''
    # bring clusters
    df, kmeans, centroids, scaler, scaled_ei_threshold_value, X = create_clusters(df)

    # Inverse scale the features so that they can later be used for modeling
    df[["avg_units_per_bldg", "ei"]] = scaler.inverse_transform(df[["avg_units_per_bldg", "ei"]])

    # When predicting a bool (emerging_market only)
    df["test_future_cluster"] = (df.sort_values(["year"])
                                .groupby(["city", "state"])[["cluster"]]
                                .shift(-2))

    df_emerging = (
        df[((df.test_future_cluster == 3) | (df.test_future_cluster == 1)) 
        & ((df.cluster == 4) | (df.cluster == 0))]
    )

    df_emerging["should_enter"] = True

    df["should_enter"] = df_emerging.should_enter

    df.should_enter = df.should_enter.fillna(False)

    df = df.merge(centroids, how="left", left_on="cluster", right_on=centroids.index)

    return df

# ------------------------ #
# Main Data Prep Functions #
# ------------------------ #

def prep_data_for_modeling_permits(df, features_for_modeling, label_feature):
    '''
    Function used to prepare the dataframe for modeling. It will read the data, take a list of features for modeling and split the data using the y as the label_feature
    '''

    # To avoid Nan's, I have removed all data from 1997 (because all the var's would be nan)
    df_model = df[df.year > 1997]

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


def permits_preprocessing_mother_function(modeling=False, features_for_modeling = [], label_feature= " "):
    """
    Main preprocessing function. Has an argument that specifies if the function is being used for modeling or not. 

    If modeling = False, then the function will acquire the data, add new features and filter for continuos cities.

    If modeling = True, then in addition to the steps above, the function will also add labels, oversample the data, split the data and scale it. It will return a train, validate, test split. 
    The function takes a list of features that will be used for modeling, as well as a label_feature, as a str, that will be used as the target variable.
    """

    if modeling == False:
        # call get_permits_model_df
        df = get_permits_model_df()

        # feature engineering
        df = add_new_features(df)

        # filter top cities
        df = filter_top_cities_building_permits(df)
        
        # df = labeling_future_data(df)


        return df

    else:
        # call get_permits_model_df
        df = get_permits_model_df()

        # feature engineering
        df = add_new_features(df)

        # filter top cities
        df = filter_top_cities_building_permits(df)

        # bring clusters
        df, kmeans, centroids, scaler, scaled_ei_threshold_value, X = create_clusters(df)

        df[["avg_units_per_bldg", "ei"]] = scaler.inverse_transform(df[["avg_units_per_bldg", "ei"]])

        # When predicting a bool (emerging_market only)
        df["test_future_cluster"] = (df.sort_values(["year"])
                                  .groupby(["city", "state"])[["cluster"]]
                                  .shift(-2))

        df_emerging = (
            df[((df.test_future_cluster == 3) | (df.test_future_cluster == 1)) 
            & ((df.cluster == 4) | (df.cluster == 0))]
        )

        df_emerging["should_enter"] = True

        df["should_enter"] = df_emerging.should_enter

        df.should_enter = df.should_enter.fillna(False)

        df = df.merge(centroids, how="left", left_on="cluster", right_on=centroids.index)

        # remove 2018 and 2019 values as they have null values
        df = df[df.year < 2018]

        #oversample the data
        df = df.append(df[df.should_enter])
        df = df.append(df[df.should_enter])

        train_scaled, validate_scaled, test_scaled, y_train, y_validate, y_test = prep_data_for_modeling_permits(df, features_for_modeling, label_feature)

        return train_scaled, validate_scaled, test_scaled, y_train, y_validate, y_test
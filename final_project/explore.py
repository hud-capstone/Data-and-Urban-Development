import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import wrangle as wr
import preprocessing as pr

import scipy.stats as stats

# --------------- #
#     Wrangle     #
# --------------- #

def print_smallest_largest_cities(df): ## Function only used in the MVP ##
    '''
    Function used to get top level understanding of min and max values
    '''
    print(
    
    f'''
        
    Morgages Max:
    The year is {df.nlargest(1, columns="final_mortgage_amount")["fiscal_year_of_firm_commitment"].values}      
    The max is ${df["final_mortgage_amount"].max():,}
    The city: {df.nlargest(1, columns="final_mortgage_amount")["project_city"].values}, {df.nlargest(1, columns="final_mortgage_amount")["project_state"].values}

    Morgages Min:
    The year is {df.nsmallest(1, columns="final_mortgage_amount")["fiscal_year_of_firm_commitment"].values}      
    The max is ${df["final_mortgage_amount"].min():,}
    The city: {df.nsmallest(1, columns="final_mortgage_amount")["project_city"].values}, {df.nsmallest(1, columns="final_mortgage_amount")["project_state"].values}


        
        
    ''')

def plot_distribution_graph_final_mortgage(df): ## Function only used in the MVP ##
    ''' 
    Function used to plot overall distribution for the final_mortgage_amount
    '''

    plt.figure(figsize=(15, 8))
    sns.distplot(df["final_mortgage_amount"])
    plt.title("What is the distribution of the final mortgage amount?")
    plt.ylabel("Frequency")
    plt.xlabel("Dollars")


# -------------- #
#  Exploration   #
# -------------- #

# What are the characteristics of loan practices in Houston (2009), Seattle (2010),  and Dallas (2012)?
def calculate_city(df, city): ## Function only used in the MVP ##
    city_data = df[df["project_city"] == city]
    city_data = city_data.set_index("fiscal_year_of_firm_commitment")
    return city_data

def plot_mortgage_volume(df, city): ## Function only used in the MVP ##
    # start with Houston 2009

    city.groupby(city.index)["final_mortgage_amount"].sum().plot.line(figsize=(15, 5))
    plt.title("Are there any spikes that we see in the total amount of dollars approved?")
    plt.xlabel("Year loan was approved")
    plt.ylabel("Dollars")

def plot_morgage_qty(df, city): ## Function only used in the MVP ##
    city.groupby(city.index)["final_mortgage_amount"].count().plot.line(figsize=(15, 5))
    plt.title("Are there any spikes that we see in the quantity of mortgage's approved?")
    plt.xlabel("Year loan was approved")
    plt.ylabel("Number of loans approved")

def plot_activity_desc(df, city): ## Function only used in the MVP ##
    # What does the activity look like the most?

    plt.figure(figsize=(15, 5))
    sns.lineplot(data=city, x=city.index, y= "final_mortgage_amount", hue="activity_description", ci=False)
    plt.title("What does the activitiy labels look like on the average amount spent?")
    plt.xlabel("Year loan was approved")
    plt.ylabel("Total amount granted, in dollars")


# Are there any trends or seasonality over time that we see in terms of number of loans approved / quantity?
def plot_quantity_over_time(df): ## Function only used in the MVP ##
    # Let's look at quantity of loans approved

    df = df.set_index("fiscal_year_of_firm_commitment")
    plt.title("What is the overall trend in terms of number of loans approved over time?")
    df.groupby(df.index)["final_mortgage_amount"].count().plot.line(figsize=(15,5))
    plt.xlabel("Year loan was approved")
    plt.ylabel("Number of loans approved")

def plot_volume_over_time(df): ## Function only used in the MVP ##
    df = df.set_index("fiscal_year_of_firm_commitment")
    plt.title("What is the overall trend in terms of number of loans approved over time?")
    df.groupby(df.index)["final_mortgage_amount"].sum().plot.line(figsize=(15,5))
    plt.xlabel("Year loan was approved")
    plt.ylabel("Number of loans approved")

def plot_desc_activity_over_time(df): ## Function only used in the MVP ##
    # Let's look at the descriptions
    df = df.set_index("fiscal_year_of_firm_commitment")
    plt.figure(figsize=(15,5))
    sns.lineplot(data=df, x=df.index, y="final_mortgage_amount", hue="activity_description", ci=False)
    plt.title("What does the activities look like over time?")
    plt.ylabel("Total amount approved, in dollars")
    plt.xlabel("Year loan was approved")


# ----------------- #
#    Clustering     #
# ----------------- #

def get_inertia(k, X):
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(X)
    return kmeans.inertia_


def visualize_clusters(df, centroids, scaled_ei_threshold_value):
    for cluster, subset in df.groupby("cluster"):
        plt.scatter(
            subset.avg_units_per_bldg, subset.ei, label="cluster" + str(cluster), alpha=0.6
        )

    centroids.plot.scatter(
        x="avg_units_per_bldg",
        y="ei",
        c="black",
        marker="x",
        s=1000,
        ax=plt.gca(),
        label="centroid",
    )

    plt.axhline(y=scaled_ei_threshold_value, color="r", linestyle='-', label="EI Threshold")
    plt.legend()
    plt.title("What groupings exist when we cluster by the average number of units per building and the evolution index?")
    plt.xlabel("Average Number of Units per Building")
    plt.ylabel("Evolution Index")
    plt.show()


def growth_rate_line_plot(df, city, year):
    

    city_df = df[df.city == city]
    
    city_df.cluster = "cluster_" + city_df.cluster.astype(str) 

    plt.figure(figsize=(15,5))
    sns.lineplot(data=city_df, x="year",y= "city_state_high_density_value_delta_pct", label="City Growth Rate")
    sns.lineplot(data=city_df, x="year", y="market_volume_delta_pct", label="Market Growth Rate")
    plt.axhline(y=0, color="r", linestyle='-', label="0% Growth")
    for i in range(22):
        plt.text(
            city_df.year.iloc[i],
            city_df.city_state_high_density_value_delta_pct.iloc[i],
            f"{city_df.cluster.iloc[i]}",
            color = 'black',
            fontsize=10,
            ha ="center"
        )
    plt.suptitle(f"What has been the trajectory of the {city_df.city.values[0]} multifamily housing market?")
    plt.title(f"""{city_df.city.values[0]} {year} Clustering Metrics: EI == {city_df[city_df.year == year].ei.values[0]:.3}; Average Number of Units per Building == {city_df[city_df.year == year].avg_units_per_bldg.values[0]:.3}""")
    plt.xlabel("Year")
    plt.ylabel("Growth Rate (%)")
    plt.legend()
    plt.show()

def plot_inertia(X):
    pd.Series({k: get_inertia(k, X) for k in range(2, 21)}).plot()
    plt.grid()
    plt.suptitle("How many clusters (k) should we create using our K-means clustering algorithm?")
    plt.title("Where do we first encounter diminishing returns with regards to intertia with subsequent increases to k?")
    plt.xlabel("k")
    plt.ylabel("inertia")
    plt.xticks(range(1, 21))
    plt.show()

# ------------------------ #
#  Clustering Exploration  #
# ------------------------ #

def mkt_value_dist_boxplot(df):
    """
    This function creates a visualization illstrating the distribution of the the greater U.S. market valuation for each cluster.
    """
    
    bp = sns.boxplot(data=df, x="cluster", y="market_volume")
    plt.title("What does the distribution of the greater U.S. market valuation as a whole look like for each cluster?")
    plt.xlabel("Cluster")
    plt.ylabel("Market Valuation")

    # divide yticks by one billion for better labels
    ylabels = ['${:.0f}'.format(x) + 'B' for x in bp.get_yticks() / 1000000000]
    bp.set_yticklabels(ylabels)
    plt.show()

def us_mkt_growth_dist_boxplot(df):
    """
    This function creates a visualization illstrating the distribution of the greater U.S. market growth rates for each cluster. 
    """

    bp = sns.boxplot(data=df, x="cluster", y="market_volume_delta_pct")
    plt.title("What is the distribution of the greater U.S. market growth rates for each cluster?")
    plt.xlabel("Cluster")
    plt.ylabel("Growth Rate (%)")

    # multiply yticks by one hundred for better labels
    ylabels = ['{:.0f}'.format(x) + '%' for x in bp.get_yticks() * 100]
    bp.set_yticklabels(ylabels)
    plt.show()

def unique_mkt_value_dist_boxplot(df):
    """
    This function creates a visualization illstrating the distribution of the market valuation for each cluster. 
    """

    bp = sns.boxplot(data=df, x="cluster", y="total_high_density_value")
    plt.title("What is the distribution of the market valuation for each cluster?")
    plt.xlabel("Cluster")
    plt.ylabel("Market Valuation")

    # divide yticks by one billion for better labels
    ylabels = ['${:.0f}'.format(x) + 'B' for x in bp.get_yticks() / 1000000000]
    bp.set_yticklabels(ylabels)
    plt.show()

def growth_rate_dist_comboplot(df):
    """
    This function creates a visualization illstrating the distribution of the market growth rates for each cluster. 
    """
    
    bp = sns.boxplot(data=df, x="cluster", y="city_state_high_density_value_delta_pct")
    sns.swarmplot(data=df, x="cluster", y="city_state_high_density_value_delta_pct", alpha=.5)
    plt.title("What is the distribution of specific market growth rates for each cluster?")
    plt.xlabel("Cluster")
    plt.ylabel("Growth Rate (%)")

    # multiply yticks by one hundred for better labels
    ylabels = ["{}".format(x) + "%" for x in bp.get_yticks()]
    bp.set_yticklabels(ylabels)
    plt.show()

def ei_dist_comboplot(df):
    """
    This function creates a visualization illstrating the distribution of the evolution index for each cluster. 
    """
    
    sns.boxplot(data=df, x="cluster", y="ei")
    sns.swarmplot(data=df, x="cluster", y="ei", alpha=.5)
    plt.title("What is the distribution of the evolution index for each cluster?")
    plt.xlabel("Cluster")
    plt.ylabel("Evolution Index")
    plt.show()

def unit_dist_boxplot(df):
    """
    This function creates a visualization illstrating the distribution of the total units for each cluster.
    """
    
    bp = sns.boxplot(data=df, x="cluster", y="total_high_density_units")
    plt.title("What is the distribution of high-density units for each cluster?")
    plt.xlabel("Cluster")
    plt.ylabel("High-Density Units")

    # divide yticks by one thousand for better labels
    ylabels = ['{:.0f}'.format(x) + 'K' for x in bp.get_yticks() / 1000]
    bp.set_yticklabels(ylabels)
    plt.show()

def building_dist_boxplot(df):
    """
    This function creates a visualization illstrating the distribution of the total buildings for each cluster.
    """
    
    sns.boxplot(data=df, x="cluster", y="total_high_density_bldgs")
    plt.title("What is the distribution of high-density buildings for each cluster?")
    plt.xlabel("Cluster")
    plt.ylabel("High-Density Buildings")
    plt.show()

def regression_pairplot(df, variables=["total_high_density_bldgs", "total_high_density_units", "total_high_density_value", "city_state_high_density_value_delta_pct", "market_volume", "market_volume_delta_pct", "ei"]):
    """
    This function creates a regression pairplot of the specified variables.
    """

    sns.pairplot(
        data=df,
        vars=variables,
        kind="reg",
        plot_kws={"line_kws": {"color": sns.color_palette("colorblind")[4]}},
        corner=True,
    )
    plt.suptitle("Regression Pairplot of Numeric Variables")
    plt.show()

# --------------------- # 
#  Statistical Testing  #
# --------------------  # 

# Is there a significant difference, by city, in the number of loans given per year? ## I think this function is only used in the MVP notebook ##

def calculate_p_values_for_num_loans(df):
    cities = df.city.unique()
    p_scores = pd.DataFrame()
    for city in cities:
        subgroup = df[df.city == city]["quantity_of_mortgages_pop"]
        tstat, p = stats.ttest_1samp(subgroup, df["quantity_of_mortgages_pop"].mean())
        result = pd.DataFrame({"city": city, "p": p}, index=[0])
        p_scores = pd.concat([p_scores, result])

    # What percentage of cities are below our alpha (significantly different)

    print(f"{p_scores[p_scores.p < 0.05].shape[0] / cities.shape[0]:.0%} of the total cities are below our alpha")

    # Of the cities with a low p values, what is the average mortgate count?
    u_cities = p_scores[p_scores.p < 0.05].city.unique()
    comparison = pd.DataFrame()
    for element in u_cities:
        score = pd.DataFrame({"city": element, "quantity_of_mortgages_pop": df[df.city == element]["quantity_of_mortgages_pop"].sum()}, index=[0])
        comparison = pd.concat([comparison, score])
        

    plt.figure(figsize=(15,5))
    graph = sns.barplot(data=comparison, x="city", y="quantity_of_mortgages_pop", palette="deep")
    graph.axhline(df["quantity_of_mortgages_pop"].mean(), ls='--', label="Mean Mortgage Count", c="black")
    plt.xticks(rotation=45, ha="right")
    plt.title("What does the count look like for cities with low p values?")
    plt.ylabel("Number of Mortgages Approved")
    plt.xlabel("City")
    plt.legend()

# Is there a significant difference, by city, in the amount of loans given per year, in dollars? ## I think this function is only used in the MVP notebook ##

def calculate_p_values_for_vol_loans(df):
    cities = df.city.unique()
    p_scores = pd.DataFrame()
    for city in cities:
        subgroup = df[df.city == city]["average_mortgage_volume_pop"]
        tstat, p = stats.ttest_1samp(subgroup, df["average_mortgage_volume_pop"].mean())
        result = pd.DataFrame({"city": city, "p": p}, index=[0])
        p_scores = pd.concat([p_scores, result])

    # What percentage of cities are below our alpha (significantly different)

    print(f"{p_scores[p_scores.p < 0.05].shape[0] / cities.shape[0]:.0%} of the total cities are below our alpha")

    # Of the cities with a low p values, what is the average mortgate count?


    u_cities = p_scores[p_scores.p < 0.05].city.unique()
    comparison = pd.DataFrame()
    for element in u_cities:
        score = pd.DataFrame({"city": element, "average_mortgage_volume_pop": df[df.city == element]["average_mortgage_volume_pop"].sum()}, index=[0])
        comparison = pd.concat([comparison, score])
        

    plt.figure(figsize=(15,5))
    graph = sns.barplot(data=comparison, x="city", y="average_mortgage_volume_pop", palette="deep")
    graph.axhline(df["average_mortgage_volume_pop"].mean(), ls='--', label="Mean Mortgage Amount", c="black")
    plt.xticks(rotation=45, ha="right")
    plt.title("What is the amount of loans approved, in dollars, for cities with low p values?")
    plt.ylabel("Mortgages Approved, in Dollars")
    plt.xlabel("City")
    plt.legend()

def plot_mortgage_vol_by_year(df): ## I think this function is only used in the MVP notebook ##
    """
    This function groups the final mortgage amount by city, state, fiscal year, and activity description and plots a line graph of the total mortgage volume.
    """

    # group by city, state, fiscal year, and activity description
    # calculate count of mortgages and volume
    city_state_all_activity = pd.DataFrame(
        df.groupby(
            [
                "project_city",
                "project_state",
                "fiscal_year_of_firm_commitment_activity",
                "activity_description",
            ]
        )["final_mortgage_amount"]
        .agg(["count", "sum"])
        .reset_index()
        .sort_values(by=["count", "sum"], ascending=False)
    )

    fig, ax = plt.subplots()

    # create lineplot
    sns.lineplot(
        data=city_state_all_activity,
        x="fiscal_year_of_firm_commitment_activity",
        y="sum",
        hue="activity_description",
        ci=False,
        ax=ax,
        # palette=palette,
    )

    # set title for legend
    legend = ax.legend()
    legend.texts[0].set_text("Activity Description")

    # format labels
    plt.xlabel("Fiscal Year")
    plt.ylabel("Total Mortgage Volume")
    plt.title("Total Mortgage Volume by Fiscal Year")
    plt.show()

def get_stat_test_results(stat, pvalue, alpha):
    "This function returns the results of a hypothesis test given the statistic, pvalue, and alpha for a test."

    print(f"statistic = {stat}")
    print(f"  p-value = {pvalue}")
    print()
    if pvalue < alpha:
        print("Reject null hypothesis")
    else:
        print("Fail to reject null hypothesis")


def high_density_rep_v_est_difference(df):
    """
    This function calculates the difference between the report and estimated numbers for the metrics relevant to
    our analysis.
    """
    
    abs_diff_buildings = (
        round(
            abs(df.five_or_more_units_bldgs_rep - df.five_or_more_units_bldgs_est).sum()
            / df.five_or_more_units_bldgs_est.sum(),
            3,
        )
        * 100
    )

    abs_diff_units = (
        round(
            abs(df.five_or_more_units_units_rep - df.five_or_more_units_units_est).sum()
            / df.five_or_more_units_units_est.sum(),
            3,
        )
        * 100
    )

    abs_diff_value = (
        round(
            abs(df.five_or_more_units_value_rep - df.five_or_more_units_value_est).sum()
            / df.five_or_more_units_value_est.sum(),
            3,
        )
        * 100
    )

    print(f"""There is an {abs_diff_buildings:.2f}% absolute difference between the reported and estimated total number of high-density, multifamily 
    buildings in the dataset.""")
    print()
    print(f"""There is an {abs_diff_units:.2f}% absolute difference between the reported and estimated total number of high-density, multifamily 
    units in the dataset.""")
    print()
    print(f"""There is an {abs_diff_value:.2f}% absolute difference between the reported and estimated total valuation of high-density, multifamily 
    structures in the dataset.""")


# ---------------- #
#   Predictions    #
# ---------------- #

def create_prediction_graph(predictions):
    '''
    This function is used to create a barplot that visualizes the number of cities within our sub category predictions
    '''
    # plt.figure(figsize=(15,5))
    sns.countplot(data=predictions, x="recommendation_label")
    plt.title("How many cities are found within each subcategory from our predictions?")

    plt.text(
        0,
        15,
        "44",
        color = 'black',
        fontsize=15,
        ha ="center"
    )

    plt.text(
        1,
        15,
        "19",
        color = 'black',
        fontsize=15,
        ha ="center"
    )

    plt.text(
        2,
        15,
        "9",
        color = 'black',
        fontsize=15,
        ha ="center"
    )

    plt.text(
        3,
        15,
        "58",
        color = 'black',
        fontsize=15,
        ha ="center"
    )

    plt.xlabel("Recommendation Label")
    plt.ylabel("Count")

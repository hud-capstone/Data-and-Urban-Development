import pandas as pd
import os

def acquire_fha_data():
    """takes fha data from hud website (url, sheet_name) turns it into pandas df"""
    
    url = 'https://www.hud.gov/sites/dfiles/Housing/documents/Initi_Endores_Firm%20Comm_DB_FY06_FY20_Q2.xlsx'
    sheet_name = "Firm Cmtmts, Iss'd and Reiss'd"

    df = pd.read_excel(url, sheet_name=sheet_name, header=6)
    
    return df

def drop_multiple_fha_numbers(df):
    """drops multiple fha_numbers by dropping issued data when a reissue is available in firm commitment activity"""
    
    #create df of rows unique fha_numbers
    unique_fha_list = df.fha_number.value_counts()[df.fha_number.value_counts() == 1]
    def in_unique_list(x):
        return x in unique_fha_list
    df_unique = df[df.fha_number.apply(in_unique_list)]
    
    #create df for rows with repeat fha_numbers
    repeat_fha_numbers = df.fha_number.value_counts()[df.fha_number.value_counts() != 1].index
    def in_repeat_list(x):
        return x in repeat_fha_numbers
    df_repeat = df[df.fha_number.apply(in_repeat_list)]
    
    #filter out "issued" mortgages, if they have been "reissued"
    df_repeat = df_repeat[df_repeat.firm_commitment_activity == 'Firm Reissued']
    
    #combine repeat and unique dataframes to show all uniques
    df = pd.concat([df_unique ,df_repeat])
    
    return df

def snake_case_column_names(df):
    """takes in a data frame, lower cases column name, replaces space with underscore and strips commas"""
    new_column_names = []
    
    for col in df.columns:
        new_column_names.append(col.lower().replace(',', "").replace(" ", "_"))
    
    df.columns = new_column_names
    return df

def change_to_bool(df):
    """takes in columns with only 2 options and turns them into booleans"""
    for col in df.columns:
        if (df[col][0] == 0) & (df[col].unique()[1] == 'Y'):
            df[col] = df[col] == "Y"
    return df


def changing_data_types(df):
    """change fha number to an object
    change year only columns to datetime"""

    df.fiscal_year_of_firm_commitment = pd.to_datetime(df.fiscal_year_of_firm_commitment, format="%Y")
    df.fiscal_year_of_firm_commitment = df.fiscal_year_of_firm_commitment.apply(lambda x: x.year)

    df.fiscal_year_of_firm_commitment_activity = pd.to_datetime(df.fiscal_year_of_firm_commitment_activity, format="%Y")
    df.fiscal_year_of_firm_commitment_activity = df.fiscal_year_of_firm_commitment_activity.apply(lambda x: x.year)

    df.fha_number = df.fha_number.astype('object')
    return df

def set_date_column(df): 
    """function drops all 2020 data and changes feature column"""
    # take this out of the function when I have an internet connection:
    df.date_of_firm_commitment_activity = pd.to_datetime(df.date_of_firm_commitment_activity)
    #drop 2020 mortgage data
    df = df[df.date_of_firm_commitment_activity < '2020-01-01']
    pd.to_datetime(df.date_of_firm_commitment_activity)
    # change date_of_firm_commitment_activity to Y only
    df.date_of_firm_commitment_activity = df.date_of_firm_commitment_activity.apply(lambda x: x.year)
    
    return df

def make_activity_construction_bool(df):
    """makes a boolean column indicating whether the mortgage was for a refinance"""
    
    df['is_refinance'] = df.activity_description == 'Refinance'
    df['is_new_construction'] = df.activity_description == 'New Construction'
    return df


def wrangle_hud():
    """
    This function does the followingL:
    1. Conditionally acquires the FHA data if the Initi_Endores_Firm Comm_DB_FY06_FY20_Q2.xlsx does not exist
    2. Reformats the column names to snake case
    3. Changes the appropriate columns to contain boolean values
    4. Changes the data types to datetime for the columns where there is only a fiscal year value 
    5. Removes the outlier whose final mortgage amount is $1
    6. Renames the project city of 55435 to Minneapolis
    7. Strips whitespace from project_city string
    8. Transform project_city string to title case
    9. Writes the transformed data to clean_data.csv
    """

    if os.path.exists('Initi_Endores_Firm Comm_DB_FY06_FY20_Q2.xlsx') == False:
        df = acquire_fha_data()
    else:
        df = pd.read_excel("Initi_Endores_Firm Comm_DB_FY06_FY20_Q2.xlsx", sheet_name="Firm Cmtmts, Iss'd and Reiss'd", header=6)

    df = snake_case_column_names(df)

    df = change_to_bool(df)

    df = changing_data_types(df)

    df = drop_multiple_fha_numbers(df)

    df = make_activity_construction_bool(df)

    # Remove outlier
    df = df[df["final_mortgage_amount"] > 10000]

    # Change name of city with just zipcode
    df["project_city"] = df["project_city"].str.replace("55435", "Minneapolis")

    # strip whitespace from project_city string
    df["project_city"] = df.project_city.str.strip()

    # transform project_city string to title case
    df["project_city"] = df.project_city.str.title()

    df.to_csv("clean_data.csv")

    return df
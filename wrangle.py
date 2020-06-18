import pandas as pd
import os

def acquire_fha_data():
    """takes fha data from hud website (url, sheet_name) turns it into pandas df"""
    
    url = 'https://www.hud.gov/sites/dfiles/Housing/documents/Initi_Endores_Firm%20Comm_DB_FY06_FY20_Q2.xlsx'
    sheet_name = "Firm Cmtmts, Iss'd and Reiss'd"

    df  = pd.read_excel(url, sheet_name=sheet_name, header=6)
    
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

def check_for_csv_file(file_name):
    if os.path.exists(file_name) == False:
        pull_csv_file()

def wrangle_hud():
    df = acquire_fha_data()

    df = snake_case_column_names(df)

    df = change_to_bool(df)
    
    df = changing_data_types(df)

    df.to_csv("clean_data.csv")

    return df
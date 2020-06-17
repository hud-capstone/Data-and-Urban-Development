import pandas as pd

def acquire_fha_data():
    """take fha data downloaded into csv and turn it into pandas df"""
    
    url = 'https://www.hud.gov/sites/dfiles/Housing/documents/Initi_Endores_Firm%20Comm_DB_FY06_FY20_Q2.xlsx'
    sheet_name = "Firm Cmtmts, Iss'd and Reiss'd"

    df  = pd.read_excel(url, sheet_name=sheet_name, header=6)
    
    return df

def snake_case_column_names(df):
    """takes in a data frame, lower cases column name, replaces space with underscore and strips commas"""
    new_column_names = []
    
    for col in df.columns:
        new_column_names.append(col.lower().strip(',').replace(" ", "_"))
    
    df.columns = new_column_names
    return df

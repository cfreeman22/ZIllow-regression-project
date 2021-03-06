# Zillow Data acquisition from the Zillow database
# importing the packages 
import env 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
 
import os 

# Initiating a Connection to the MYSQL server with connection info from env.py

def get_db_url(db_name):
    from env import username, host, password
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'



def get_zillow_data():
    '''
    This function reads csv stored in the computer or the Zillow data from the Codeup db into a dataframe.
    '''
    filename = "zillow_df.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        # read the SQL query into a dataframe
        sql_query = """
                SELECT parcelid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt,yearbuilt, taxamount, fips, logerror, transactiondate FROM properties_2017
                JOIN predictions_2017 using(parcelid)
                JOIN propertylandusetype USING(propertylandusetypeid)
                WHERE propertylandusedesc IN ("Single Family Residential",                       
                                  "Inferred Single Family Residential")
                """
        df = pd.read_sql(sql_query, get_db_url('zillow')) #SQL query , database name, Pandas df

        # Write that dataframe to disk for later. Called "caching" the data for later.
        # Return the dataframe to the calling code
        # renaming column names to one's I like better
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value',
                              'yearbuilt':'year_built'})
        df.to_csv(filename) 

        # Return the dataframe to the calling code
        # renaming column names to one's I like better
         
        return df  
#libs
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


#------------------------------------------------------------------------------
def data_check(data):
    print("------------------- Shape -------------------------------")  
    print(data.shape) 
    print("__________________________________________________________") 
    print("------------------- Head ---------------------------------")  
    print(data.head()) 
    print("__________________________________________________________") 
    print("--------------------Info -------------------------------")
    print(data.info())
    print("__________________________________________________________") 
    print("--------------------- Describe --------------------------")
    print(data.describe().T )
    print("__________________________________________________________") 
    print("---------------------- Null ----------------------")   
    print(data.isnull().sum())
    print("__________________________________________________________") 
    
    
#--------------------------------------------------------------------------
def grab_columns_names(data, cat_th=10, car_th=20):  
    cat_cols = [col for col in data.columns if str(data[col].dtypes) in ["category","object","bool"] ] 
    #If int and float variables have less than 10 unique values, these variables are categorical
    num_but_cat = [col for col in  data.columns if data[col].nunique() < 10 and data[col].dtypes in ["int","float"]]  
    #cardinal
    cat_but_car = [col for col in data.columns if data[col].nunique() > 20 and str(data[col].dtypes) in ["int","float64"]] 
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in data.columns if data[col].dtypes in ["int","float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    return cat_cols, num_cols, cat_but_car
  
  
 #------------------------------------------------------------------------------------------------

def numerical_summary(data,numerical_col, plot = False):  
    quantiles = [0.05 ,0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90] 
    print(data[numerical_col].describe(quantiles).T) 
    
    print("_________________________________________________________________________________")  
    if plot: 
        data[numerical_col].hist() 
        plt.xlabel(numerical_col)
        plt.title(numerical_col) 
        plt.show(block=True) 
    #usage numerical_summary(data,"age",plot=True)

#---------------------------------------------------------------------------------------------
def cat_summary(data , column_name, plot = False): 
    print(pd.DataFrame({column_name:data[column_name].value_counts(), 
                      "Ratio":100 * data[column_name].value_counts() / len(data)})) 

    print("_________________________________________________________________________________")    
    if plot: 
        sns.countplot(x=data[column_name],data=data)
        plt.show(block=True)
    #cat_summary(data , "sex", plot= True)

#--------------------------------------------------------------------
def outliers_thresholds(data, column_name, q1=0.25, q3=0.75):
    quartile1 = data[column_name].quantile(q1)
    quartile3 = data[column_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit , up_limit
    #outliers_thresholds(data,"salary")
    
#--------------------------------------------------------------------

def check_outlier(data, column_name):
    low_limit, up_limit = outliers_thresholds(data, column_name)
    # aslında yukarıada yaptığımız any yani bool ile herhangi bir boş aykırı değer var mı sorusuna denk gelir
    if data[(data[column_name]>up_limit) | (data[column_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    #check_outlier(data,"age")
    
 #--------------------------------------------------------------------
def grab_outliers(data, column_name, index = False):
    low,up = outliers_thresholds(data, column_name)
    
    if data[(data[column_name]<low) | (data[column_name] > up)].shape[0] > 10:
        print(data[((data[column_name] < low) | (data[column_name] > up))].head())
    else:
        print(data[((data[column_name] < low) | (data[column_name]> up))])

    if index:
        outlier_index = data[((data[column_name]< low) | (data[column_name] > up))].index
        return outlier_index
    #grab_outliers(data,"column1",True)

#---------------------------------------------------------------------------------------
def remove_outliers(data,column_name):
    low_limit, up_limit = outliers_thresholds(data,column_name)
    data_without_outliers = data[~((data[column_name] < low_limit) | (data[column_name] > up_limit))]
    return data_without_outliers
    #for col in numerical_cols:
        #new_df = remove_outliers(data,col)
        
#---------------------------------------------------------------------------------------
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] > up_limit),variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit),variable] = low_limit
    #replace_with_thresholds(data,"col2")
 
#-----------------------------------------------------------------------------------------

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
        #missing_values = missing_values_table(app_train)
        #missing_values.head(20)

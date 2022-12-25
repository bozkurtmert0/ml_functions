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
    
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"] ] 
    #If int and float variables have less than 10 unique values, these variables are categorical
    num_but_cat = [col for col in  df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int","float"]]  
    #cardinal
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["int","float64"]] 
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_cat]
    num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    return cat_cols, num_cols, cat_but_car
  
  
 #------------------------------------------------------------------------------------------------

def numerical_summary(data,numerical_col, plot = False):  
    quantiles = [0.05 ,0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90] 
    print(dataframe[numerical_col].describe(quantiles).T) 
    
    print("_________________________________________________________________________________")  
    if plot: 
        dataframe[numerical_col].hist() 
        plt.xlabel(numerical_col)
        plt.title(numerical_col) 
        plt.show(block=True) 
#num_summary(df,"age",plot=True)

#---------------------------------------------------------------------------------------------
def cat_summary(dataframe , col_name, plot = False): 
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(), 
                      "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)})) 

    print("_________________________________________________________________________________")    
    if plot: 
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)
    #cat_summary(df , "sex", plot= True)

#--------------------------------------------------------------------

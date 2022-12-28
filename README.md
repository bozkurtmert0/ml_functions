# Functions 
[Data Preprocessing](https://github.com/bozkurtmert0/ml_functions/blob/main/data_preprocessing.py)
---------------------------------------------------------
* data_check
* grab_columns_names
* numerical_summary
* cat_summary
* outliers_thresholds
* check_outlier
* grab_outliers
* remove_outliers
* replace_with_thresholds
-----------------------------------------------------------
### data_check(data)

**argument**:
* **data**: pandas DataFrame

**output**:
* *.shape*  -----> return a tuple representing the dimensionality of the DataFrame.
* *.head()* -----> returns the first 5 rows for the DataFrame.
* *.info()* -----> prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.
* *.describe()*  ------> Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
* *.isnull().sum()* -----> Detects and sums the missing values of each column.
-----------------------------------------------------------
### grab_columns_names(data, cat_th, car_th)
**arguments** 
 * **data**: pandas DataFrame
 * **cat_th** : int value. If int and float variables have less than ***cat_th*** unique values, these variables are categorical.
 * **car_th** : int value.  If int and float variables have more than ***car_th*** unique values, these variables are cardinal.
 
**return**: cat_cols, num_cols, cat_but_car
* **cat_cols**: List of Categorical columns.
* **num_cols**: List of Numerical columns.
* **cat_but_car**: List of Cardinal columns. Like *age* variables, its is numerical but u can also use like categorical.
-----------------------------------------------------------

import pandas as pd 
from collections import OrderedDict

def descriptive_stats(df):
    # Segregate Numerical columns and Categorical columns

  numerical_col = df.select_dtypes(exclude = "object").columns
  categorical_col = df.select_dtypes(include = "object").columns

  # Checking Stats: Numerical Columns
  num_stats = []
  cat_stats = []
  data_info = []

  for i in numerical_col:

    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    LWR = Q1 - 1.5*IQR
    UWR = Q3 + 1.5*IQR

    outlier_count = len(df[(df[i] < LWR) | (df[i] > UWR)])
    outlier_percentage = (outlier_count / len(df)) * 100

    numericalstats = OrderedDict({
        "Feature":i,
        "Mean":df[i].mean(),
        "Median":df[i].median(),
        "Minimum":df[i].min(),
        "Maximum":df[i].max(),
        "Q1":Q1,
        "Q3":Q3,
        "IQR":IQR,
        "LWR":LWR,
        "UWR":UWR,
        "Outlier Count":outlier_count,
        "Outlier Percentage":outlier_percentage,
        "Standard Deviation":df[i].std(),
        "Variance":df[i].var(),
        "Skewness":df[i].skew(),
        "Kurtosis":df[i].kurtosis()
        })
    num_stats.append(numericalstats)
    numerical_stats_report = pd.DataFrame(num_stats)

  # Checking for Categorical columns
  for i in categorical_col:
    cat_stats1 = OrderedDict({
        "Feature":i,
        "Unique Values":df[i].nunique(),
        "Value Counts":df[i].value_counts().to_dict(),
        "Mode":df[i].mode()[0]
    })
    cat_stats.append(cat_stats1)
    categorical_stats_report = pd.DataFrame(cat_stats)

  # Checking datasetinformation
  for i in df.columns:
    data_info1 = OrderedDict({
        "Feature":i,
        "Data Type":df[i].dtype,
        "Missing_Values":df[i].isnull().sum(),
        "Unique_Values":df[i].nunique(),
        "Value_Counts":df[i].value_counts().to_dict()
        })
    data_info.append(data_info1)
    data_info_report = pd.DataFrame(data_info)

  return numerical_stats_report,categorical_stats_report,data_info_report


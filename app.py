# Import Data Manipulation Libraries
import numpy as np 
import pandas as pd 

# Import scikit learn libraries
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# Import Regression Model
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict


# Step1: Data Ingestion:
def data_ingestion():
    return pd.read_csv(r'C:\EnergyConsumption_PredictionModel\train_energy_data.csv')

# Step2: Data Exploration
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


# Step3: Data Preprocessing
def data_preprocessing(df):
    X = df.drop(columns = 'Energy Consumption',axis = 1)
    y = df['Energy Consumption']

  # Split the Dataset into train and test

    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state = 10)

  # Use Encoding Technique to convert all categorical columns into numerical columns

  # Note: Train data is always fit_transform() and Test always transform()

    categorical_col = X.select_dtypes(include = 'object').columns

    from sklearn.preprocessing import LabelEncoder
    for i in categorical_col:
        le = LabelEncoder()
        X_train[i] = le.fit_transform(X_train[i])  # Seen Data
        X_test[i] = le.transform(X_test[i])        # Unseen Data

  # Using Normalization Technique
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)          # Seen Data
    X_test = sc.transform(X_test)                # Unseen Data
    return X_train,X_test,y_train,y_test

# Step4: Model Building
def model_building(df):
    rf = RandomForestRegressor().fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    r2score = r2_score(y_test,y_pred)
    return r2score

# Function Calling
df = data_ingestion()

num_stats,cat_stats,data_info = descriptive_stats(df)

X_train,X_test,y_train,y_test = data_preprocessing(df)

rf = model_building(df)

print("The Model Accuracy is : ",rf)
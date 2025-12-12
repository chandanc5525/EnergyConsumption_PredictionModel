from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_data(df, target_col, test_size=0.3, random_state=10):
    X = df.drop(columns=[target_col])
    y = df[target_col]


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state)


    # Encoding
    categorical_col = X.select_dtypes(include="object").columns
    for i in categorical_col:
        le = LabelEncoder()
        X_train[i] = le.fit_transform(X_train[i])
        X_test[i] = le.transform(X_test[i])


    # Scaling
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    return X_train, X_test, y_train, y_test
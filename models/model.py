from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def build_model(X_train, y_train):
    model = RandomForestRegressor().fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)
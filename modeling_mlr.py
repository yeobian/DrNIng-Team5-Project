"""
File: modeling_mlr.py
Purpose: Train and evaluate Multiple Linear Regression model.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def run_mlr(df):
    # Example: Predict 'Life expectancy' using all other features
    X = df.drop(columns=["Life expectancy"])
    y = df["Life expectancy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R²:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

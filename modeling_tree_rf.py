"""
File: modeling_tree_rf.py
Purpose: Train and evaluate Decision Tree and Random Forest models.
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def run_tree_rf(df):
    X = df.drop(columns=["Life expectancy"])
    y = df["Life expectancy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    print("Decision Tree R²:", r2_score(y_test, dt_pred))
    print("Decision Tree RMSE:", mean_squared_error(y_test, dt_pred, squared=False))

    # Random Forest
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("Random Forest R²:", r2_score(y_test, rf_pred))
    print("Random Forest RMSE:", mean_squared_error(y_test, rf_pred, squared=False))

    # Feature importance
    importances = rf_model.feature_importances_
    features = X.columns
    plt.barh(features, importances)
    plt.title("Feature Importance (Random Forest)")
    plt.show()

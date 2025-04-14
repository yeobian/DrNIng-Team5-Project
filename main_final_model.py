"""
File: main_final_model.py
Purpose: Run full pipeline: cleaning, EDA, and modeling.
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

# %%
df = pd.read_csv('cleaned_data.csv')
df
# %%
df.info()
# %%
df.describe()
# %%
import matplotlib.pyplot as plt

# Ensure only numeric columns are used for the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])

# Generate a correlation matrix
correlation_matrix = numeric_df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# Show the heatmap
plt.title("Correlation Heatmap")
plt.show()

#%%
label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country'])
country_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Country Encoding Mapping:", country_mapping)
df.head()
# %%
# Convert country_mapping into a DataFrame
country_mapping_df = pd.DataFrame(list(country_mapping.items()), columns=['Country', 'Encoded Value'])

# Display the DataFrame
print(country_mapping_df)
# %%
df.info()
# %%
# Sort the DataFrame by the 'year' column
df = df.sort_values(by='year')

# Display the first few rows of the sorted DataFrame
df.head()

# %%
# Divide the DataFrame into training and testing sets based on the year
train_df = df[df['year'] <= 2022]
test_df = df[df['year'] > 2022]

# Separate features (X) and target (y) for training and testing sets
X_train = train_df.drop(columns=['Life expectancy at birth, total (years)'])
y_train = train_df['Life expectancy at birth, total (years)']

X_test = test_df.drop(columns=['Life expectancy at birth, total (years)'])
y_test = test_df['Life expectancy at birth, total (years)']

# Display the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# %%
# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
# %%
# Initialize the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model on the training data
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Regressor - Mean Squared Error (MSE):", mse_dt)
print("Decision Tree Regressor - R-squared (R2):", r2_dt)
# %%
# Initialize the Bayesian Ridge Regressor
br_model = BayesianRidge()

# Train the model on the training data
br_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_br = br_model.predict(X_test)

# Evaluate the model
mse_br = mean_squared_error(y_test, y_pred_br)
r2_br = r2_score(y_test, y_pred_br)

print("Bayesian Ridge Regressor - Mean Squared Error (MSE):", mse_br)
print("Bayesian Ridge Regressor - R-squared (R2):", r2_br)


# %%

# %%

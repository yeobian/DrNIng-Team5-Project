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
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('cleaned_data.csv')
df
# %%
df.info()
# %%
df.describe()
# %%
numeric_df = df.select_dtypes(include=[np.number])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

plt.title("Correlation Heatmap")
plt.show()

#%%
label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country'])
country_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Country Encoding Mapping:", country_mapping)
df.head()
# %%
country_mapping_df = pd.DataFrame(list(country_mapping.items()), columns=['Country', 'Encoded Value'])

print(country_mapping_df)
# %%
df.info()
# %%
df = df.sort_values(by='year')

df.head()

# %%
train_df = df[df['year'] <= 2022]
test_df = df[df['year'] > 2022]

X_train = train_df.drop(columns=['Life expectancy at birth, total (years)'])
y_train = train_df['Life expectancy at birth, total (years)']

X_test = test_df.drop(columns=['Life expectancy at birth, total (years)'])
y_test = test_df['Life expectancy at birth, total (years)']

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# %%
rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse_rfr = mean_squared_error(y_test, y_pred)
r2_rfr = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse_rfr)
print("R-squared (R2):", r2_rfr)
# %%
dt_model = DecisionTreeRegressor(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Regressor - Mean Squared Error (MSE):", mse_dt)
print("Decision Tree Regressor - R-squared (R2):", r2_dt)
# %%
br_model = BayesianRidge()

br_model.fit(X_train, y_train)

y_pred_br = br_model.predict(X_test)

mse_br = mean_squared_error(y_test, y_pred_br)
r2_br = r2_score(y_test, y_pred_br)

print("Bayesian Ridge Regressor - Mean Squared Error (MSE):", mse_br)
print("Bayesian Ridge Regressor - R-squared (R2):", r2_br)


# %%
results = {
    "Model": ["Random Forest Regressor", "Decision Tree Regressor", "Bayesian Ridge Regressor"],
    "Mean Squared Error (MSE)": [mse_rfr, mse_dt, mse_br],
    "R-squared (R2)": [r2_rfr, r2_dt, r2_br]
}

results_df = pd.DataFrame(results)

print(results_df)
# %%
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="R-squared (R2)", data=results_df, palette="viridis")

plt.title("R-squared (R2) Scores by Model", fontsize=16)
plt.xlabel("Model", fontsize=12)
plt.ylabel("R-squared (R2)", fontsize=12)

plt.ylim(0.999, 1.0)

for index, row in results_df.iterrows():
    plt.text(index, row["R-squared (R2)"], 
             f'{row["R-squared (R2)"]:.6f}', 
             ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="Mean Squared Error (MSE)", data=results_df, palette="viridis")

plt.title("Mean Squared Error (MSE) Scores by Model", fontsize=16)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)

for index, row in results_df.iterrows():
    plt.text(index, row["Mean Squared Error (MSE)"], 
             f'{row["Mean Squared Error (MSE)"]:.2f}', 
             ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
# %%

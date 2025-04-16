"""
File: main_final_model.py
Purpose: Run full pipeline: cleaning, EDA, and modeling.
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
# Check missing values
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.show()

# %%
numeric_df = df.select_dtypes(include=[np.number])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# =========================
# New EDA Plots
# =========================

# %%
# Distribution of life expectancy
plt.figure(figsize=(8, 5))
sns.histplot(df['Life expectancy at birth, total (years)'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Life Expectancy")
plt.xlabel("Life Expectancy at Birth (Years)")
plt.tight_layout()
plt.show()

# %%
# Top 10 countries with highest average life expectancy
top_life = df.groupby('country')['Life expectancy at birth, total (years)'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_life.values, y=top_life.index, palette="crest")
plt.title("Top 10 Countries by Average Life Expectancy")
plt.xlabel("Average Life Expectancy")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# %%
# GDP per capita vs Life Expectancy
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='GDP per capita (current US$)', y='Life expectancy at birth, total (years)', alpha=0.6)
plt.title("GDP per Capita vs Life Expectancy")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Pairplot for key variables
key_vars = ['GDP per capita (current US$)', 'Death rate, crude (per 1,000 people)',
            'Population ages 65 and above (% of total population)',
            'Life expectancy at birth, total (years)']

sns.pairplot(df[key_vars], kind='scatter', diag_kind='kde')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.tight_layout()
plt.show()

# =========================
# Preprocessing Before Modeling
# =========================

# %%
df = df.drop(columns=["Life expectancy at birth, female (years)", "Life expectancy at birth, male (years)"])

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

#%%
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

# Evaluate
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

# Display the results in the same format
print("KNN Regressor - Mean Squared Error (MSE):", mse_knn)
print("KNN Regressor - R-squared (R2):", r2_knn)

# %%
# Import missing numpy module due to environment reset
import numpy as np
import matplotlib.pyplot as plt

# Prepare the existing performance results including KNN
model_names = ["Random Forest", "Decision Tree", "Bayesian Ridge", "KNN Regressor"]
rmse_values = [0.862, 1.326, 1.853, np.sqrt(0.551)]  # KNN RMSE from MSE
r2_values = [0.991, 0.980, 0.961, 0.994]

# Plotting RMSE and R² comparison
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# RMSE plot
axs[0].bar(model_names, rmse_values, color='cornflowerblue')
axs[0].set_title("RMSE of Regression Models")
axs[0].set_ylabel("Root Mean Squared Error")
axs[0].set_ylim(0, max(rmse_values) + 0.5)

# R² plot
axs[1].bar(model_names, r2_values, color='mediumseagreen')
axs[1].set_title("R² of Regression Models")
axs[1].set_ylabel("R² Score")
axs[1].set_ylim(0.9, 1.01)

plt.tight_layout()
plt.show()

# %%
# Feature importance from Random Forest Regressor
feature_importances = rf_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importance
print(importance_df)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")

plt.title("Feature Importance from Random Forest Regressor", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)

plt.tight_layout()
plt.show()



# %%
# =========================
# Time Series Modeling: ARIMA and SARIMAX
# =========================

print("\n--- Time Series Modeling (ARIMA/SARIMAX) ---")
# %%
# Create a country-level average time series
ts_df = df.groupby('year')['Life expectancy at birth, total (years)'].mean().reset_index()
ts_df.columns = ['year', 'life_expectancy']

# Set index to year for time series
ts_df.set_index('year', inplace=True)
# %%
# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(ts_df, marker='o', linestyle='-')
plt.title("Average Global Life Expectancy Over Time")
plt.ylabel("Life Expectancy")
plt.xlabel("Year")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# ARIMA Model
# =========================
# %%
arima_model = ARIMA(ts_df, order=(1, 1, 1))  # ARIMA(p,d,q)
arima_result = arima_model.fit()

# Forecast the next 5 years
arima_forecast = arima_result.forecast(steps=5)
print("ARIMA Forecast (next 5 years):")
print(arima_forecast)
# %%
# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(ts_df, label="Observed")
plt.plot(range(ts_df.index[-1] + 1, ts_df.index[-1] + 6), arima_forecast, label="Forecast", marker='o')
plt.title("ARIMA Forecast for Global Life Expectancy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# SARIMAX Model
# =========================
# %%
sarimax_model = SARIMAX(ts_df, order=(1, 1, 1), seasonal_order=(0, 1, 1, 4))
sarimax_result = sarimax_model.fit(disp=False)

# Forecast next 5 years
sarimax_forecast = sarimax_result.forecast(steps=5)
print("SARIMAX Forecast (next 5 years):")
print(sarimax_forecast)
# %%
# Plot SARIMAX forecast
plt.figure(figsize=(10, 5))
plt.plot(ts_df, label="Observed")
plt.plot(range(ts_df.index[-1] + 1, ts_df.index[-1] + 6), sarimax_forecast, label="SARIMAX Forecast", marker='x')
plt.title("SARIMAX Forecast for Global Life Expectancy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# =========================
# Prophet
# =========================

print("\n--- Time Series Modeling (Prophet) ---")
# %%
from prophet import Prophet

# Prepare data for Prophet
df_prophet = ts_df.reset_index()
df_prophet.columns = ['ds', 'y']  # Prophet requires these column names

# Create and fit the model
prophet_model = Prophet()
prophet_model.fit(df_prophet)

# Create future dataframe and forecast
future = prophet_model.make_future_dataframe(periods=5, freq='Y')
prophet_forecast = prophet_model.predict(future)

# Show forecasted values
print("Prophet Forecast (next 5 years):")
print(prophet_forecast[['ds', 'yhat']].tail(5))
# %%
# Plot the forecast
fig = prophet_model.plot(prophet_forecast)
plt.title("Prophet Forecast for Global Life Expectancy")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# =========================
# Exponential Smoothing (ETS)
# =========================

print("\n--- Time Series Modeling (Exponential Smoothing / ETS) ---")
# %%
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit ETS model with additive trend
ets_model = ExponentialSmoothing(ts_df, trend='add', seasonal=None)
ets_result = ets_model.fit()

# Forecast next 5 years
ets_forecast = ets_result.forecast(steps=5)
print("ETS Forecast (next 5 years):")
print(ets_forecast)
# %%
# Plot the ETS forecast
plt.figure(figsize=(10, 5))
plt.plot(ts_df, label="Observed")
plt.plot(range(ts_df.index[-1] + 1, ts_df.index[-1] + 6), ets_forecast, label="ETS Forecast", marker='s')
plt.title("Exponential Smoothing (ETS) Forecast for Global Life Expectancy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

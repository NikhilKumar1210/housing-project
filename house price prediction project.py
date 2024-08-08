# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from local CSV
df = pd.read_csv('housing.csv')

# Check for non-numeric data
print(df.info())

# If there are any categorical columns, convert them using one-hot encoding
# In the California Housing dataset, the 'ocean_proximity' column is categorical
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['median_house_value'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Feature Engineering
df['PRICE_PER_ROOM'] = df['median_house_value'] / df['total_rooms']

# Preparing Data for Modeling
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building and Evaluation
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)
print('Linear Regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_lin_reg)))
print('Linear Regression R^2:', r2_score(y_test, y_pred_lin_reg))

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)
print('Decision Tree RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print('Decision Tree R^2:', r2_score(y_test, y_pred_dt))

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
print('Random Forest RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print('Random Forest R^2:', r2_score(y_test, y_pred_rf))

gb_reg = GradientBoostingRegressor()
gb_reg.fit(X_train, y_train)
y_pred_gb = gb_reg.predict(X_test)
print('Gradient Boosting RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_gb)))
print('Gradient Boosting R^2:', r2_score(y_test, y_pred_gb))

# Insights and Recommendations
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", feature_importances)

sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()
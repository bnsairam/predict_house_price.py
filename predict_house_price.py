import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Sample housing data (replace with real dataset or API data)
data = {
    'size_sqft': np.random.uniform(800, 4000, 1000),  # House size in square feet
    'bedrooms': np.random.randint(1, 6, 1000),        # Number of bedrooms
    'age': np.random.randint(0, 50, 1000),            # Age of house in years
    'location_score': np.random.uniform(1, 10, 1000), # Location quality (1-10)
    'price': np.random.uniform(100000, 1000000, 1000) # Price in USD
}
df = pd.DataFrame(data)

# Add some realistic correlation to price
df['price'] = (df['size_sqft'] * 200 + df['bedrooms'] * 50000 - df['age'] * 2000 + 
               df['location_score'] * 30000 + np.random.normal(0, 50000, 1000))

# Prepare features (X) and target (y)
X = df[['size_sqft', 'bedrooms', 'age', 'location_score']]
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('price_prediction.png')
plt.show()

# Feature importance visualization (based on coefficients)
feature_importance = pd.Series(np.abs(model.coef_), index=X.columns)
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance in House Price Prediction')
plt.xlabel('Coefficient Magnitude')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Bonus: Example code to load data from a CSV or mock API (uncomment to use)
"""
import requests
def fetch_housing_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        df_api = pd.DataFrame(data)
        return df_api
    return None

# Example usage
# api_url = 'https://example.com/housing-data-api'
# housing_data = fetch_housing_data(api_url)
# if housing_data is not None:
#     X_new = housing_data[['size_sqft', 'bedrooms', 'age', 'location_score']]
#     predicted_prices = model.predict(X_new)
#     print(f'Predicted prices for new houses: {predicted_prices}')
"""

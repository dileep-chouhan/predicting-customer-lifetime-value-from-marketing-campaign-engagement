import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_customers = 500
data = {
    'Campaign_A_Engagement': np.random.randint(0, 10, num_customers),  # Engagement score (0-9)
    'Campaign_B_Engagement': np.random.randint(0, 10, num_customers),
    'Campaign_C_Engagement': np.random.randint(0, 10, num_customers),
    'Website_Visits': np.random.randint(1, 20, num_customers),
    'Purchase_Amount': np.random.randint(0, 1000, num_customers) # Total purchase amount
}
df = pd.DataFrame(data)
#Adding a feature representing Customer Lifetime Value (CLV) - this is what we aim to predict
df['CLV'] = df['Purchase_Amount'] + (df['Campaign_A_Engagement'] + df['Campaign_B_Engagement'] + df['Campaign_C_Engagement']) * 50 + df['Website_Visits'] * 10
# --- 2. Data Preparation and Modeling ---
# Define features (X) and target (y)
X = df.drop('CLV', axis=1)
y = df['CLV']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# --- 3. Model Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 4. Visualization ---
# Visualize actual vs. predicted CLV
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red') #Line of perfect prediction
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.title('Actual vs. Predicted Customer Lifetime Value')
plt.grid(True)
plt.tight_layout()
output_filename = 'actual_vs_predicted_clv.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Feature Importance visualization (optional but recommended for a complete analysis)
feature_importance = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(10,6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance')
plt.ylabel('Coefficient')
plt.tight_layout()
output_filename = 'feature_importance.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, housing.target, train_size=0.8, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R² Score (Coefficient of Determination): {r2:.4f}')
explainer = shap.TreeExplainer(model)
sample_size = 500
X_train_sample = X_train.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_train_sample)

shap.summary_plot(shap_values, X_train_sample)

# Create scatter plots for each feature
features = X_train.columns.tolist()
n_features = len(features)
n_cols = 4
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()

for i, feature in enumerate(features):
    # Correct way to create SHAP dependence plot
    shap.dependence_plot(
        feature,
        shap_values,
        X_train_sample,
        ax=axes[i],
        show=False
    )
    axes[i].set_title(f"SHAP Dependence: {feature}", fontsize=12)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("SHAP Dependence Plots for All Features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
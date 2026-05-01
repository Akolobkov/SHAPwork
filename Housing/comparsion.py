from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostRegressor
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, kendalltau
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, housing.target, train_size=0.8, random_state=42)

model1 = CatBoostRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='RMSE',
    eval_metric='R2',
    l2_leaf_reg=3.0,
    random_strength=0.5,
    bagging_temperature=0.5,
    early_stopping_rounds=50,
    od_type='Iter',
    thread_count=-1,
    random_seed=42,
    verbose=100,
    bootstrap_type='Bayesian',
    min_child_samples=10,
    rsm=0.8
)
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)


sample_size = 500
X_train_sample = X_train.sample(n=sample_size, random_state=42)
explainer1 = shap.TreeExplainer(model1)
shap_values_Catboost = explainer1(X_train_sample)
model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

explainer = shap.TreeExplainer(model2)

X_train_sample = X_train.sample(n=sample_size, random_state=42)
explainer2 = shap.TreeExplainer(model2)
shap_values_RF = explainer2(X_train_sample)
mean_shap_1 = np.abs(shap_values_Catboost.values).mean(axis=0)
mean_shap_2 = np.abs(shap_values_RF.values).mean(axis=0)

spearman_corr, _ = spearmanr(mean_shap_1, mean_shap_2)
kendall_corr, _ = kendalltau(mean_shap_1, mean_shap_2)
print(spearman_corr, kendall_corr)
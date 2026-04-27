from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, kendalltau
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


model = CatBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
explainer = shap.TreeExplainer(model)
sample_size = 500
X_train_sample = X_train.sample(n=sample_size, random_state=42)
shap_values = explainer(X_train_sample)
shap_values_class1_Catboost = shap_values
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
explainer = shap.TreeExplainer(model)

X_train_sample = X_train.sample(n=sample_size, random_state=42)
shap_values = explainer(X_train_sample)
shap_values_class1_RF = shap_values[:, :, 1]
mean_shap_1 = np.abs(shap_values_class1_Catboost.values).mean(axis=0)
mean_shap_2 = np.abs(shap_values_class1_RF.values).mean(axis=0)

spearman_corr, _ = spearmanr(mean_shap_1, mean_shap_2)
kendall_corr, _ = kendalltau(mean_shap_1, mean_shap_2)
print(spearman_corr, kendall_corr)
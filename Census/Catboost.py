
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np

X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


model = CatBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Точность модели:', accuracy)
explainer = shap.TreeExplainer(model)
sample_size = 500
X_train_sample = X_train.sample(n=sample_size, random_state=42)
shap_values = explainer(X_train_sample)
print(shap_values.shape)
shap_values_class1 = shap_values
shap.plots.beeswarm(shap_values_class1)
shap.plots.heatmap(shap_values_class1)
features_to_plot = ["Age", "Education-Num", "Hours per week", "Capital Gain"]
n_features = len(features_to_plot)
n_cols = 4
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()
for i, feature in enumerate(features_to_plot):
    shap.plots.scatter(shap_values_class1[:, feature], show=False, ax=axes[i])
    axes[i].set_title(f"SHAP зависимость: {feature}", fontsize=12)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Графики зависимости SHAP для всех признаков", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
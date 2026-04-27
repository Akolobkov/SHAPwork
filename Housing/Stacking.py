from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np


housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, housing.target, train_size=0.8, random_state=42)


rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

cb_model = CatBoostRegressor(
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


print("Обучение Random Forest...")
rf_model.fit(X_train, y_train)

print("\nОбучение CatBoost...")
cb_model.fit(X_train, y_train)


rf_train_pred = rf_model.predict(X_train).reshape(-1, 1)
rf_test_pred = rf_model.predict(X_test).reshape(-1, 1)

cb_train_pred = cb_model.predict(X_train).reshape(-1, 1)
cb_test_pred = cb_model.predict(X_test).reshape(-1, 1)


X_train_meta = np.hstack([rf_train_pred, cb_train_pred])
X_test_meta = np.hstack([rf_test_pred, cb_test_pred])


meta_model = LinearRegression()
meta_model.fit(X_train_meta, y_train)


y_pred_meta = meta_model.predict(X_test_meta)


mse = mean_squared_error(y_test, y_pred_meta)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred_meta)
r2 = r2_score(y_test, y_pred_meta)

print('\n' + '=' * 50)
print("РЕЗУЛЬТАТЫ СТЕКИНГ МОДЕЛИ:")
print('=' * 50)
print(f'Средняя ошибка модели (MAE): {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R^2 - score (Коэффициент детерминации): {r2:.4f}')



def shap_stacking_predict(X_input):
    """
    Функция для SHAP анализа стекинг модели
    """
    if isinstance(X_input, pd.DataFrame):
        X_input = X_input.values

    # Предсказания базовых моделей
    rf_pred = rf_model.predict(X_input).reshape(-1, 1)
    cb_pred = cb_model.predict(X_input).reshape(-1, 1)

    # Мета-признаки
    X_meta = np.hstack([rf_pred, cb_pred])

    # Финальное предсказание
    final_pred = meta_model.predict(X_meta)

    return final_pred


# Создание explainer для стекинг модели
print("\nЗапуск SHAP анализа для стекинг модели...")

# Используем меньшую выборку для SHAP (для скорости)
sample_size = 500
X_train_sample = X_train.sample(n=sample_size, random_state=42)

# Создаем маску для признаков
background_data = X_train_sample.values

# Используем Explainer с пользовательской функцией предсказания
explainer = shap.Explainer(shap_stacking_predict, background_data)

# Расчет SHAP значений (может занять некоторое время)
print("Расчет SHAP значений...")
shap_values = explainer(X_train_sample[:100])  # Для скорости берем 100 объектов

# 5. Визуализация SHAP для стекинг модели

# Глобальная важность признаков
print("\nГлобальная важность признаков (SHAP):")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_sample[:100], show=False)
plt.title("SHAP значения для стекинг модели (Random Forest + CatBoost)")
plt.tight_layout()
plt.show()

# Барплот с важностью признаков
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_sample[:100], plot_type="bar", show=False)
plt.title("Важность признаков в стекинг модели")
plt.tight_layout()
plt.show()

# 6. Графики зависимости SHAP для каждого признака
features = X.columns.tolist()
n_features = len(features)
n_cols = 4
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()

for i, feature in enumerate(features):
    # Создаем dependence plot для каждого признака
    shap.dependence_plot(
        feature,
        shap_values.values,
        X_train_sample[:100],
        ax=axes[i],
        show=False
    )
    axes[i].set_title(f"SHAP зависимость: {feature}", fontsize=12)

# Удаляем лишние подграфики
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Графики зависимости SHAP для стекинг модели (RF + CatBoost)",
             fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# 7. Дополнительный анализ: вклад каждой базовой модели
print("\n" + "=" * 50)
print("АНАЛИЗ ВКЛАДА БАЗОВЫХ МОДЕЛЕЙ:")
print("=" * 50)
print(f"Вес Random Forest в стекинге: {meta_model.coef_[0]:.4f}")
print(f"Вес CatBoost в стекинге: {meta_model.coef_[1]:.4f}")
print(f"Интерцепт (свободный член): {meta_model.intercept_:.4f}")

rf_pred_test = rf_model.predict(X_test)
cb_pred_test = cb_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred_test)
cb_mae = mean_absolute_error(y_test, cb_pred_test)

print("\nСравнение моделей (MAE):")
print(f"Random Forest: {rf_mae:.4f}")
print(f"CatBoost: {cb_mae:.4f}")
print(f"Stacking: {mae:.4f}")

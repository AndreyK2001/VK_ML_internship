import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_auc_score

train = pd.read_csv("./data/train_df.csv")
test = pd.read_csv("./data/test_df.csv")

X_train = train.drop(columns=["search_id", "target"])
y_train = train["target"].values

X_test = test.drop(columns=["search_id", "target"])
y_test = test["target"].values

# Определяем константные признаки в обучающем наборе данных
constant_columns_train = [col for col in X_train.columns if X_train[col].nunique() == 1]

# Проверяем константные признаки в тестовом наборе данных
constant_columns_test = [col for col in X_test.columns if X_test[col].nunique() == 1]

# Пересечение константных признаков в обучающем и тестовом наборе данных
constant_features = list(set(constant_columns_train) & set(constant_columns_test))

# Изучаем категориальные признаки (будем считать категориальными те признаки, которые имеют меньше 10 уникальных значений)
categorical_features_train = [
    col for col in X_train.columns if 1 < X_train[col].nunique() <= 10
]
categorical_features_test = [
    col for col in X_test.columns if 1 < X_test[col].nunique() <= 10
]

# Пересечение категориальных признаков в обучающем и тестовом наборе данных
categorical_features = list(
    set(categorical_features_train) & set(categorical_features_test)
)


# Создаем объект OneHotEncoder, указывая drop='first' для избежания ловушки фиктивных переменных
encoder = OneHotEncoder(drop="first", sparse_output=False)

# Обучаем энкодер на категориальных признаках обучающего набора данных и преобразуем их
encoded_cats_train = encoder.fit_transform(X_train[categorical_features])

# Преобразуем категориальные признаки тестового набора данных
encoded_cats_test = encoder.transform(X_test[categorical_features])

# Посмотрим на размерность полученных массивов после кодирования
encoded_cats_train.shape, encoded_cats_test.shape

# Сначала удалим из исходных данных константные и категориальные признаки
non_categorical_features = list(
    set(X_train.columns) - set(categorical_features) - set(constant_features)
)

# Формируем исходные некатегориальные данные для обучающего и тестового наборов данных
non_cat_train_df = X_train[non_categorical_features]
non_cat_test_df = X_test[non_categorical_features]

# Создаем DataFrame из закодированных категориальных данных для обучающего и тестового наборов
encoded_cats_train_df = pd.DataFrame(
    encoded_cats_train, columns=encoder.get_feature_names_out(categorical_features)
)
encoded_cats_test_df = pd.DataFrame(
    encoded_cats_test, columns=encoder.get_feature_names_out(categorical_features)
)

# Сбрасываем индексы, чтобы избежать проблем при конкатенации
non_cat_train_df.reset_index(drop=True, inplace=True)
encoded_cats_train_df.reset_index(drop=True, inplace=True)
non_cat_test_df.reset_index(drop=True, inplace=True)
encoded_cats_test_df.reset_index(drop=True, inplace=True)

# Конкатенируем некатегориальные и закодированные категориальные данные
final_train_df = pd.concat([non_cat_train_df, encoded_cats_train_df], axis=1)
final_test_df = pd.concat([non_cat_test_df, encoded_cats_test_df], axis=1)


scaler = StandardScaler()
scaled_train_df = scaler.fit_transform(final_train_df)
scaled_test_df = scaler.transform(final_test_df)


X_train_const = sm.add_constant(scaled_train_df)
model = sm.Logit(y_train, X_train_const).fit(disp=0)


p_values = model.pvalues[1:]  # исключаем константу

# Определяем статистически не значимые признаки
insignificant_feature_indices = np.where(p_values > 0.05)[0]

# Удаление не значимых признаков из X_train и X_test
X_train_reduced = np.delete(scaled_train_df, insignificant_feature_indices, axis=1)
X_test_reduced = np.delete(scaled_test_df, insignificant_feature_indices, axis=1)


# Функция для вычисления NDCG на основе фактических и предсказанных рейтингов
def calculate_ndcg(y_true, y_score):
    true_relevance = np.asarray([y_true])
    scores = np.asarray([y_score])
    ndcg = ndcg_score(true_relevance, scores)
    return ndcg


clf = LogisticRegression()
clf.fit(X_train_reduced[:-500], y_train[:-500])
predictions_val = clf.predict_proba(X_train_reduced[-500:])[:, 1]

ndcg_val = calculate_ndcg(y_train[-500:], predictions_val)
ROC_val = roc_auc_score(y_train[-500:], predictions_val)
print(f"val NDCG: {ndcg_val:.4}; val ROC-AUC: {ROC_val:.4}")


predictions = clf.predict_proba(X_test_reduced)[:, 1]
ndcg_test = calculate_ndcg(y_test, predictions)
ROC_test = roc_auc_score(y_test, predictions)
print(f"test NDCG: {ndcg_test:.4}; test ROC-AUC: {ROC_test:.4}")

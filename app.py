import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)

st.set_page_config(page_title="Wine Quality", page_icon="🍷", layout="wide")
APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_config():
    with open(APP_DIR / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / "test.csv")


@st.cache_data
def get_test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
test = load_test()

feature_cols = [col for col in test.columns if col != "target"]
default_threshold = float(config["threshold"])

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)

missing_test_cols = [col for col in feature_cols if col not in test.columns]

if missing_test_cols:
    st.error(f"В test.csv не хватает столбцов: {missing_test_cols}")
    st.stop()

X_test = test[feature_cols]
y_test = test["target"]
y_proba = get_test_proba(model, X_test)


st.title("Оценка качества вина")

threshold = st.sidebar.slider(
    "Порог классификации",
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.01,
    key="threshold"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Оценка вина",
    "Модельные метрики",
    "Пакетное предсказание",
    "Что выучила модель"
])

with tab1:
    st.header("Оценка одного вина")

    values = {}

    for col in feature_cols:
        if col == "is_red":
            is_red = st.toggle("Красное вино", value=False)
            values[col] = int(is_red)
        else:
            min_value = float(test[col].min())
            max_value = float(test[col].max())
            median_value = float(test[col].median())

            if min_value == max_value:
                values[col] = min_value
            else:
                if col == "density":
                    values[col] = st.slider(
                        col,
                        min_value=min_value,
                        max_value=max_value,
                        value=median_value,
                        step=0.0001,
                        format="%.5f",
                        key=f"wine_{col}"
                    )
                else:
                    values[col] = st.slider(
                        col,
                        min_value=min_value,
                        max_value=max_value,
                        value=median_value,
                        key=f"wine_{col}"
                    )

    if st.button("Оценить вино"):
        wine = pd.DataFrame([values], columns=feature_cols)
        probability = model.predict_proba(wine)[0, 1]
        prediction = int(probability >= threshold)

        st.metric("Вероятность хорошего вина", f"{probability:.1%}")

        if prediction == 1:
            st.success("Модель считает вино хорошим")
        else:
            st.warning("Модель не считает вино хорошим")

with tab2:
    st.header("Модельные метрики")

    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    st.subheader("Матрица ошибок")

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Не хорошее", "Хорошее"])
    ax.set_yticklabels(["Не хорошее", "Хорошее"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("F1", f"{f1:.3f}")

    tn, fp, fn, tp = cm.ravel()

    current_fpr = fp / (fp + tn) if fp + tn else 0
    current_tpr = tp / (tp + fn) if tp + fn else 0

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    st.subheader("ROC-кривая")

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.scatter(
        current_fpr,
        current_tpr,
        color="red",
        label=f"Текущий порог = {threshold:.2f}"
    )
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    st.subheader("PR-кривая")

    fig, ax = plt.subplots()
    ax.plot(
        pr_recall,
        pr_precision,
        label=f"PR-AUC = {pr_auc:.3f}"
    )
    ax.scatter(
        recall,
        precision,
        color="red",
        label=f"Текущий порог = {threshold:.2f}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Комментарий для сомелье")

    if threshold > default_threshold:
        st.info(
            "При увеличении порога модель будет реже относить вина "
            "к хорошим. Подборка станет строже: обычно precision "
            "увеличивается, а recall снижается."
        )
    elif threshold < default_threshold:
        st.info(
            "При уменьшении порога модель будет чаще относить вина "
            "к хорошим. Подборка станет шире: обычно recall "
            "увеличивается, а precision может снизиться."
        )
    else:
        st.info("Используется порог по умолчанию")

with tab3:
    st.header("Пакетное предсказание")

    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл с признаками вин",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            batch = pd.read_csv(uploaded_file)

            missing_cols = [
                col for col in feature_cols
                if col not in batch.columns
            ]

            if missing_cols:
                st.error(
                    f"В файле не хватает обязательных столбцов: "
                    f"{', '.join(missing_cols)}"
                )
            else:
                X_batch = batch[feature_cols]

                if X_batch.isnull().any().any():
                    st.error("В файле есть пропущенные значения.")
                else:
                    try:
                        probabilities = model.predict_proba(X_batch)[:, 1]
                        predictions = (probabilities >= threshold).astype(int)

                        result = batch.copy()
                        result["probability"] = probabilities
                        result["prediction"] = predictions
                        result["decision"] = np.where(
                            predictions == 1,
                            "Хорошее",
                            "Не хорошее"
                        )

                        st.dataframe(result, use_container_width=True)

                        csv = result.to_csv(index=False).encode("utf-8-sig")

                        st.download_button(
                            "Скачать результаты",
                            data=csv,
                            file_name="wine_predictions.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Не удалось обработать файл: {e}")

        except Exception as e:
            st.error(f"Не удалось прочитать CSV-файл: {e}")

with tab4:
    st.header("Какие признаки влияют на оценку вина")

    if hasattr(model, "named_steps"):
        classifier = list(model.named_steps.values())[-1]
    else:
        classifier = model

    if not hasattr(classifier, "coef_"):
        st.error(
            "Не удалось получить коэффициенты: "
            "последний шаг модели не является обученной логистической регрессией."
        )
    else:
        coefficients = pd.Series(
            classifier.coef_.ravel(),
            index=feature_cols,
            name="Коэффициент"
        ).sort_values()

        st.markdown(
            "Коэффициенты логистической регрессии после масштабирования "
            "признаков. Положительное значение повышает оценку в пользу "
            "хорошего вина, отрицательное — в пользу не хорошего"
        )

        fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(coefficients))))
        colors = [
            "tomato" if value < 0 else "seagreen"
            for value in coefficients.values
        ]
        ax.barh(coefficients.index, coefficients.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Коэффициент логистической регрессии")
        ax.set_ylabel("Признак")
        ax.set_title("Вклад признаков в предсказание класса «хорошее вино»")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        strongest_negative = coefficients.head(3)
        strongest_positive = coefficients.tail(3).sort_values(ascending=False)

        col_negative, col_positive = st.columns(2)

        with col_negative:
            st.subheader("Сильнее всего против хорошего вина")
            for feature, value in strongest_negative.items():
                st.write(f"**{feature}**: {value:.3f}")

        with col_positive:
            st.subheader("Сильнее всего за хорошее вино")
            for feature, value in strongest_positive.items():
                st.write(f"**{feature}**: +{value:.3f}")

        st.caption(
            "Чем дальше коэффициент от нуля по модулю, тем сильнее "
            "связь признака с предсказанием модели"
        )

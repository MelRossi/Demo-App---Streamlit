import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: flex-end;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([8, 2])
with col2:
    st.image("Logo.png", width=180) 

# Funciones utilitarias
def cargar_datos(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file, encoding='latin-1')
        return data
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def realizar_grid_search(estimator, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def mostrar_grafico(data, column_x, column_y, plot_type):
    plt.figure(figsize=(10, 6))
    if plot_type == "Scatterplot":
        sns.scatterplot(data=data, x=column_x, y=column_y, hue=column_y, palette="viridis")
        plt.title(f"Scatterplot entre {column_x} y {column_y}")
    elif plot_type == "Heatmap":
        contingency_table = pd.crosstab(data[column_x], data[column_y])
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f"Heatmap (tabla de contingencia) entre {column_x} y {column_y}")
    elif plot_type == "Histograma":
        sns.histplot(data[column_x], kde=True, bins=20, color="blue", label=column_x)
        sns.histplot(data[column_y], kde=True, bins=20, color="orange", label=column_y)
        plt.legend()
        plt.title(f"Histogramas de {column_x} y {column_y}")
    elif plot_type == "Boxplot":
        sns.boxplot(data=data, x=column_x, y=column_y)
        plt.title(f"Boxplot entre {column_x} y {column_y}")
    st.pyplot(plt)
    plt.clf()

# Configuración de la app
st.markdown('<h1 style="color:#EA7072;">NeuroData Lab</h1>', unsafe_allow_html=True)
st.write("Modelo Predictivo Clínico")

# Paso 1: Carga de datos
st.header("1. Cargar Datos")
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    data = cargar_datos(uploaded_file)
    if data is not None:
        st.write("Vista previa de los datos cargados:")
        st.dataframe(data.head())

        # Comparación Gráfica
        st.header("Comparación Gráfica")
        st.write("Selecciona dos columnas para comparar y el tipo de gráfico a visualizar:")

        col1, col2 = st.columns(2)
        with col1:
            column_x = st.selectbox("Selecciona la primera columna (X):", data.columns, key="col_x")
        with col2:
            column_y = st.selectbox("Selecciona la segunda columna (Y):", data.columns, key="col_y")

        plot_type = st.selectbox("Selecciona el tipo de gráfico:", ["Scatterplot", "Heatmap", "Histograma", "Boxplot"])

        if column_x and column_y:
            mostrar_grafico(data, column_x, column_y, plot_type)

        # Selección de la variable objetivo
        st.header("2. Selección de Columnas")
        target_col = st.selectbox("Selecciona la variable objetivo (Y):", data.columns)
        feature_cols = st.multiselect("Selecciona las características (X):", [col for col in data.columns if col != target_col])

        if target_col and feature_cols:
            st.header("3. Entrenamiento del Modelo")
            X = data[feature_cols]
            y = data[target_col]

            X = pd.get_dummies(X, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

            # Modelos y parámetros
            param_grid_lr = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
            param_grid_dt = {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'criterion': ['gini', 'entropy']}
            param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}

            best_params_lr = realizar_grid_search(LogisticRegression(random_state=42), param_grid_lr, X_train_res, y_train_res)
            best_params_dt = realizar_grid_search(DecisionTreeClassifier(random_state=42), param_grid_dt, X_train_res, y_train_res)
            best_params_rf = realizar_grid_search(RandomForestClassifier(random_state=42), param_grid_rf, X_train_res, y_train_res)

            st.write("Selecciona el modelo de aprendizaje:")
            model_choice = st.selectbox("Modelo:", ["Logistic Regression", "Decision Tree", "Random Forest"])

            if model_choice == "Logistic Regression":
                modelo = LogisticRegression(**best_params_lr, max_iter=1000, random_state=42)
            elif model_choice == "Decision Tree":
                modelo = DecisionTreeClassifier(**best_params_dt, random_state=42)
            elif model_choice == "Random Forest":
                modelo = RandomForestClassifier(**best_params_rf, random_state=42)

            modelo.fit(X_train_res, y_train_res)

            y_pred = modelo.predict(X_test_scaled)
            y_prob = modelo.predict_proba(X_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            st.write("**Exactitud del modelo:**", accuracy)

            if len(np.unique(y_test)) > 2:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                st.write("**AUC-ROC (multiclase):**", roc_auc)
            else:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                st.write("**AUC-ROC:**", roc_auc)

                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Curva ROC")
                plt.legend(loc="lower right")
                st.pyplot(plt)

            st.write("**Matriz de Confusión:**")
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.xlabel("Predicción")
            plt.ylabel("Verdadero")
            st.pyplot(plt)

            st.text("Reporte de Clasificación:")
            st.text(classification_report(y_test, y_pred))

            st.header("4. Predicción")
            predict_file = st.file_uploader("Archivo de predicción (CSV):", type=["csv"], key="predict")

            if predict_file:
                predict_data = cargar_datos(predict_file)
                if predict_data is not None:
                    st.write("Datos cargados para predicción:")
                    st.dataframe(predict_data.head())

                    predict_data = pd.get_dummies(predict_data, drop_first=True)
                    predict_data = predict_data.reindex(columns=X.columns, fill_value=0)

                    predictions = modelo.predict(predict_data)
                    probabilities = modelo.predict_proba(predict_data)

                    st.write("**Resultados de las predicciones:**")
                    result_df = predict_data.copy()
                    result_df["Predicción"] = predictions
                    result_df["Probabilidad"] = probabilities.max(axis=1)
                    st.dataframe(result_df)

                    st.download_button(
                        label="Descargar resultados",
                        data=result_df.to_csv(index=False).encode('utf-8'),
                        file_name="resultados_prediccion.csv",
                        mime="text/csv"
                    )

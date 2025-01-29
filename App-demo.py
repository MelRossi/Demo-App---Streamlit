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
    st.image("Logo.png", width=120)
    
st.markdown("""
    <style>
    .gradient-text {
        font-size: 48px; /* Tamaño más grande para el título */
        font-weight: bold;
        background: linear-gradient(to right, #776BDC, #EB373A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .subtitle {
        font-size: 14px; /* Tamaño más pequeño para descripción */
        font-weight: normal;
        color: #666; /* Color gris más tenue */
        text-align: center;
        margin-top: -20px; /* Reduce la separación entre el título y el subtítulo */
    }
    </style>
    <h1 class="gradient-text">NeuroData Lab</h1>
    <p class="subtitle">Modelo predictivo clínico basado en inteligencia artificial</p>
""", unsafe_allow_html=True)


# Paso 1: Carga de datos
st.write("## <span style='color: #EA937F;'>1. Cargar Datos</span>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# Validar que el archivo fue cargado antes de continuar
if uploaded_file is not None:
    data = cargar_datos(uploaded_file)

    if data is not None:  # Verifica que la carga fue exitosa
        st.write("Vista previa de los datos cargados:")
        st.dataframe(data.head())
    else:
        st.error("No se pudieron cargar los datos. Verifica el archivo e intenta nuevamente.")
        st.stop()
else:
    st.warning("Por favor, sube un archivo CSV para continuar.")
    st.stop()

# Comparación Gráfica
st.write("## <span style='color: #EA937F; font-size: 24px; '>Comparación Gráfica</span>", unsafe_allow_html=True)
st.write("Selecciona dos columnas para comparar y el tipo de gráfico a visualizar:")

col1, col2 = st.columns(2)
with col1:
    column_x = st.selectbox("Selecciona la primera columna (X):", data.columns, key="col_x")
with col2:
    column_y = st.selectbox("Selecciona la segunda columna (Y):", data.columns, key="col_y")

plot_type = st.selectbox("Selecciona el tipo de gráfico:", ["Scatterplot", "Heatmap", "Histograma", "Boxplot"])

if column_x and column_y:
    mostrar_grafico(data, column_x, column_y, plot_type)

    # Generar conclusiones basadas en el tipo de gráfico
    st.write("## <span style='color: #EA937F; font-size: 24px;'>Conclusión</span>", unsafe_allow_html=True)

    if plot_type == "Scatterplot":
        st.write(f"El gráfico de dispersión entre **{column_x}** y **{column_y}** ayuda a visualizar la relación entre ambas variables. "
                 "Si se observa una tendencia clara (ascendente o descendente), podría indicar una correlación significativa. "
                 "Si los puntos están dispersos sin un patrón, la relación puede ser débil o inexistente.")

    elif plot_type == "Heatmap":
        st.write(f"El heatmap muestra la distribución conjunta entre **{column_x}** y **{column_y}**. "
                 "Valores más altos en ciertas áreas indican una mayor concentración de datos. "
                 "Esto puede ayudar a identificar patrones en la relación entre ambas variables.")

    elif plot_type == "Histograma":
        st.write(f"Los histogramas permiten analizar la distribución de las variables **{column_x}** y **{column_y}**. "
                 "Si la forma de la distribución es sesgada o bimodal, esto puede indicar diferencias en la naturaleza de los datos "
                 "o la presencia de subgrupos dentro de la muestra.")

    elif plot_type == "Boxplot":
        st.write(f"El boxplot entre **{column_x}** y **{column_y}** permite visualizar la dispersión y presencia de valores atípicos. "
                 "Si hay una gran cantidad de valores fuera de los bigotes del boxplot, es posible que existan outliers significativos.")

# Selección de la variable objetivo
st.write("## <span style='color: #EA937F;'>2. Selección de Columnas</span>", unsafe_allow_html=True)
target_col = st.selectbox(
    "Variable objetivo (Y):",
    data.columns,
    index=data.columns.get_loc("RESPUESTA_BINARIA") if "RESPUESTA_BINARIA" in data.columns else 0
)
feature_cols = st.multiselect("Selecciona las características (X):", [col for col in data.columns if col != target_col])

if target_col and feature_cols:
    st.write("## <span style='color: #EA937F;'>3. Entrenamiento del Modelo</span>", unsafe_allow_html=True)
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

    st.write("## <span style='color: #EA937F; font-size: 24px; '>Selecciona el modelo de aprendizaje:</span>", unsafe_allow_html=True)
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
           
    # Agregar conclusión basada en los resultados
    st.write("## <span style='color: #EA937F; font-size: 24px;'>Conclusión</span>", unsafe_allow_html=True)
    st.write("El modelo ha sido entrenado exitosamente y evaluado en los datos de prueba. La exactitud obtenida indica el rendimiento del modelo en la clasificación de los datos. Se recomienda revisar la matriz de confusión y la curva ROC para analizar el comportamiento de las predicciones. Dependiendo del objetivo clínico, se pueden ajustar los hiperparámetros o probar modelos adicionales para mejorar los resultados.")

    st.write("## <span style='color: #EA937F;'>4. Predicción</span>", unsafe_allow_html=True)
    predict_file = st.file_uploader("Archivo de predicción (CSV):", type=["csv"], key="predict")

    if predict_file:
        predict_data = cargar_datos(predict_file)
        if predict_data is not None:
            st.write("## <span style='color: #EA937F; font-size: 24px; '>Datos cargados para predicción:</span>", unsafe_allow_html=True)
            st.dataframe(predict_data.head())

            predict_data = pd.get_dummies(predict_data, drop_first=True)
            predict_data = predict_data.reindex(columns=X.columns, fill_value=0)

            predictions = modelo.predict(predict_data)
            probabilities = modelo.predict_proba(predict_data)

            st.write("## <span style='color: #EA937F; font-size: 24px; '>**Resultados de las predicciones:**</span>", unsafe_allow_html=True)
            result_df = predict_data.copy()
            result_df["Predicción"] = predictions
            result_df["Probabilidad"] = probabilities.max(axis=1)
            st.dataframe(result_df)

            # Crear gráfico solo si hay más de una clase predicha
            fig, ax = plt.subplots()

            pred_counts = result_df["Predicción"].value_counts()

            if len(pred_counts) > 1:
                pred_counts.plot(kind="bar", ax=ax, color=["#08306B", "#4292C6"])
                ax.set_title("Distribución de Predicciones")
                ax.set_xlabel("Clase Predicha")
                ax.set_ylabel("Frecuencia")
                st.pyplot(fig)
            else:
                st.warning("Todas las predicciones pertenecen a una sola clase. Puede ser necesario ajustar los datos o el modelo.")


            st.download_button(
                label="Descargar resultados",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name="resultados_prediccion.csv",
                mime="text/csv"
            )

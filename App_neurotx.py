import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

def realizar_grid_search(modelo, parametros, X_train, y_train):
    grid_search = GridSearchCV(modelo, parametros, cv=5, scoring='accuracy', n_jobs=-1)
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

@st.cache_data
def cargar_datos_entrenamiento():
    return pd.read_csv("/mnt/data/dftrain.csv", encoding="latin-1")
   
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
    <h1 class="gradient-text">Neuro Tx Latam</h1>
    <p class="subtitle">Modelo predictivo clínico basado en inteligencia artificial</p>
""", unsafe_allow_html=True)


# Paso 1: Carga de datos
st.write("## <span style='color: #EA937F;'>1. Cargar Datos</span>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding="latin-1")
    st.write("Archivo cargado exitosamente.")
else:
    st.warning("No se subió un archivo. Se usará un dataset predeterminado.")
    data = pd.read_csv("dataset.csv", encoding="latin-1")

st.write("Vista previa de los datos:")
st.dataframe(data.head())

# Inicializar column_x y column_y como None para evitar NameError
column_x, column_y = None, None

# Verificar que los datos están cargados antes de continuar
if "data" not in locals() or data is None or data.empty:
    st.error("Error: No se han cargado datos. Por favor, sube un archivo CSV antes de continuar.")
    st.stop()

# Obtener columnas numéricas y categóricas
numeric_columns = data.select_dtypes(include=['number']).columns
categorical_columns = data.select_dtypes(exclude=['number']).columns

# Seleccionar el tipo de gráfico
plot_type = st.selectbox("Selecciona el tipo de gráfico:", ["Scatterplot", "Heatmap", "Histograma", "Boxplot"])

# Asegurar que column_x y column_y solo se definan si hay columnas disponibles
if plot_type == "Histograma" and len(numeric_columns) > 0:
    column_x = st.selectbox("Selecciona una variable numérica para el histograma:", numeric_columns, key="col_hist")

elif plot_type == "Scatterplot" and len(numeric_columns) > 1:
    col1, col2 = st.columns(2)
    with col1:
        column_x = st.selectbox("Selecciona la primera columna (X):", numeric_columns, key="col_x")
    with col2:
        column_y = st.selectbox("Selecciona la segunda columna (Y):", numeric_columns, key="col_y")

elif plot_type == "Heatmap":
    col1, col2 = st.columns(2)
    with col1:
        column_x = st.selectbox("Selecciona la primera columna (X):", data.columns, key="col_x")
    with col2:
        column_y = st.selectbox("Selecciona la segunda columna (Y):", data.columns, key="col_y")

elif plot_type == "Boxplot" and len(categorical_columns) > 0 and len(numeric_columns) > 0:
    col1, col2 = st.columns(2)
    with col1:
        column_x = st.selectbox("Selecciona la variable categórica (X):", categorical_columns, key="col_x")
    with col2:
        column_y = st.selectbox("Selecciona la variable numérica (Y):", numeric_columns, key="col_y")

# Verificar que column_x esté definido antes de usarlo
if column_x is not None:
    st.write("## <span style='color: #EA937F; font-size: 24px;'>Gráfico</span>", unsafe_allow_html=True)

    if plot_type == "Histograma":
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column_x], kde=True, bins=20, color="blue")
        plt.title(f"Histograma de {column_x}")
        plt.xlabel(column_x)
        plt.ylabel("Frecuencia")
        st.pyplot(plt)
    elif column_y:
        mostrar_grafico(data, column_x, column_y, plot_type)

   # Generar conclusiones basadas en los datos y el tipo de gráfico
st.write("## <span style='color: #EA937F; font-size: 24px;'>Conclusión</span>", unsafe_allow_html=True)

if plot_type == "Scatterplot":
    correlacion = data[column_x].corr(data[column_y])
    if correlacion > 0.7:
        conclusion = f"Existe una fuerte correlación positiva ({correlacion:.2f}) entre **{column_x}** y **{column_y}**."
    elif correlacion < -0.7:
        conclusion = f"Existe una fuerte correlación negativa ({correlacion:.2f}) entre **{column_x}** y **{column_y}**."
    else:
        conclusion = f"No se observa una correlación significativa ({correlacion:.2f}) entre **{column_x}** y **{column_y}**."
    st.write(conclusion)

elif plot_type == "Heatmap":
    tabla_contingencia = pd.crosstab(data[column_x], data[column_y])
    conclusion = f"El heatmap muestra la distribución de **{column_x}** y **{column_y}**, sugiriendo que ciertas combinaciones ocurren con mayor frecuencia."
    st.write(conclusion)

elif plot_type == "Histograma":
    sesgo_x = data[column_x].skew()
    conclusion_x = f"**{column_x}** tiene una distribución {'sesgada a la derecha' if sesgo_x > 0.5 else 'sesgada a la izquierda' if sesgo_x < -0.5 else 'simétrica'} (sesgo = {sesgo_x:.2f})."
    st.write(conclusion_x)

elif plot_type == "Boxplot":
    # Verificar que column_y es numérica antes de calcular los outliers
    if column_y in numeric_columns:
        if data[column_y].isna().sum() > 0:
            st.warning(f"La variable **{column_y}** contiene valores nulos. Los resultados pueden no ser precisos.")

        # Calcular outliers usando IQR (rango intercuartílico)
        q1 = data[column_y].quantile(0.25)
        q3 = data[column_y].quantile(0.75)
        iqr = q3 - q1  # Rango intercuartílico
        outliers_y = ((data[column_y] < q1 - 1.5 * iqr) | (data[column_y] > q3 + 1.5 * iqr)).sum()
        
        st.write(f"**{column_y}** tiene {outliers_y} valores atípicos detectados.")
    else:
        st.warning(f"La variable **{column_y}** no es numérica. No se pueden calcular valores atípicos.")

# **Selección de la variable objetivo**
st.write("## <span style='color: #EA937F;'>2. Entrenar de Modelo</span>", unsafe_allow_html=True)
st.write("## <span style='color: #EA937F; font-size: 24px;'>Modelo elegido -> Random Forest</span>", unsafe_allow_html=True)
# Carga archivo de entrenamiento
data2 = pd.read_csv("dftrain.csv", encoding="latin-1")  

st.write("Vista previa del segundo dataset:")
st.dataframe(data2.head())

# Verificar si la columna RESPUESTA_BINARIA existe en el dataset
if "RESPUESTA_BINARIA" in data.columns:
    X = data2.drop(columns=["RESPUESTA_BINARIA"])
    y = data2["RESPUESTA_BINARIA"]

    smote = SMOTE(sampling_strategy=0.4, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    st.write("Distribución después del balanceo:", y_resampled.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Exactitud del modelo:** {accuracy:.4f}")

    roc_auc = roc_auc_score(y_test, y_prob)
    st.write(f"**AUC-ROC:** {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdadero")
    ax.set_title("Matriz de Confusión")
    st.pyplot(fig)

    st.text("Reporte de Clasificación:")
    reporte = classification_report(y_test, y_pred, output_dict=True)
    df_reporte = pd.DataFrame(reporte).transpose()
    st.table(df_reporte)

else:
    st.error("La columna 'RESPUESTA_BINARIA' no está en el dataset. Por favor, revisa los datos.")

 # Agregar conclusión basada en los resultados
st.write("## <span style='color: #EA937F; font-size: 24px;'>Descripción</span>", unsafe_allow_html=True)
st.write("""Métricas de evaluación:\n
- Precisión (Precision): De todas las predicciones positivas realizadas por el modelo, ¿cuántas fueron realmente correctas?\n
- Recall (Sensibilidad): De todos los casos positivos reales, ¿cuántos fueron correctamente identificados por el modelo?\n
- F1-score: Media armónica entre precisión y recall. Ofrece un equilibrio entre precisión y recall.
- Accuracy (Exactitud): Del total de predicciones realizadas, ¿cuántas fueron correctas? Mide el rendimiento general del modelo.
- Support (Soporte): Número de muestras en cada clase. Indica cuántos ejemplos reales hay de cada clase.
- Macro avg (Promedio macro): Promedio no ponderado de las métricas (precisión, recall, F1) para cada clase.
- Weighted avg (Promedio ponderado): Promedio ponderado de las métricas para cada clase, donde los pesos son el soporte (número de muestras en cada clase).""")

joblib.dump(rf_model, "rfc_model.pkl")

st.write("## <span style='color: #EA937F;'>3. Predicción</span>", unsafe_allow_html=True)
predict_file = st.file_uploader("Archivo de predicción (CSV):", type=["csv"], key="predict")

if predict_file:
    predict_data = cargar_datos(predict_file)

    if predict_data is not None and not predict_data.empty:
        st.write("## <span style='color: #EA937F; font-size: 24px;'>Datos cargados para predicción:</span>", unsafe_allow_html=True)
        st.dataframe(predict_data.head())

        def procesar_nuevo_dataset(predict_data, dftrain):
            """
            Asegura que el nuevo dataset tenga las mismas columnas y formatos que el dataset de referencia.
            """
            columnas_referencia = dftrain.columns.tolist()
            
            # Convertir columnas categóricas a numéricas como en el dataset de referencia
            predict_data = pd.get_dummies(predict_data)

            # Asegurar que las columnas coincidan exactamente con las del modelo entrenado
            missing_cols = set(X.columns) - set(predict_data.columns)
            extra_cols = set(predict_data.columns) - set(X.columns)

            # Agregar las columnas faltantes con valor 0
            for col in missing_cols:
                predict_data[col] = 0

            # Eliminar columnas extra que no están en el modelo entrenado
            predict_data = predict_data[X.columns]

            st.write(f"🔹 Columnas agregadas: {missing_cols}")
            st.write(f"🔹 Columnas eliminadas: {extra_cols}")

            return predict_data

        # Procesar el dataset para que coincida con el modelo entrenado
        predict_data = procesar_nuevo_dataset(predict_data, X)

        # Realizar predicciones
        try:
            predictions = rf_model.predict(predict_data)
            probabilities = rf_model.predict_proba(predict_data)

            result_df = predict_data.copy()
            result_df["Predicción"] = predictions
            result_df["Probabilidad"] = probabilities.max(axis=1)

            st.write("## Resultados de las predicciones:")
            st.dataframe(result_df)

        except Exception as e:
            st.error(f"Se produjo un error al realizar las predicciones: {e}")
            st.stop()

        # Crear gráfico de distribución de predicciones
        fig, ax = plt.subplots()
        pred_counts = result_df["Predicción"].value_counts()

        if len(pred_counts) > 1:
            pred_counts.plot(kind="bar", ax=ax, color=["#08306B", "#4292C6"])
            ax.set_title("Distribución de Predicciones")
            ax.set_xlabel("Clase Predicha")
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Todas las predicciones pertenecen a una sola clase. Puede ser necesario ajustar los datos o el modelo.")

else:
    st.error("El archivo de predicción está vacío o no se pudo procesar.")

#  Cargar el modelo entrenado
modelo = joblib.load("rfc_model.pkl")
        
# Barra lateral para ingresar valores
st.sidebar.header(" Ingrese valores para la predicción")

columnas = ["EDAD", "SEXO", "TUMOR_PRIMARIO", "SUBTIPO_HISTOLOGICO",
            "No._METS", "TAMAÃ\x91O_(mm)", "LOCALIZACION", "DOSIS_(Gy)",
            "TECNICA", "TRATAMIENTO_SISTEMICO"]

@st.cache_data  # Cachear la función para mayor eficiencia
def cargar_opciones(data2, columnas):
    opciones = {}
    for col in columnas:
        if col in data2.columns:
            if data2[col].dtype == 'object':
                # Para columnas categóricas, convertir a enteros si es posible
                try:
                    opciones[col] = [int(x) for x in data2[col].unique().tolist()]
                except ValueError:
                    # Si no se puede convertir a entero, mantener como string
                    opciones[col] = data2[col].unique().tolist()
            else:
                opciones[col] = {
                    "min": int(data2[col].min()),
                    "max": int(data2[col].max()),
                    "mean": int(data2[col].mean())
                }
    return opciones

opciones = cargar_opciones(data2, columnas)

# Diccionario para almacenar los datos del usuario
datos_usuario = {}

# Iterar sobre las columnas para crear los widgets de la barra lateral
for col in columnas:
    if col in opciones:
        if isinstance(opciones[col], list):  # Columna categórica o convertida a enteros
            # Si los valores son enteros, usar un slider
            if all(isinstance(x, int) for x in opciones[col]):
                valor = st.sidebar.slider(f"{col}", min(opciones[col]), max(opciones[col]), int(sum(opciones[col]) / len(opciones[col])), step=1)
            else:  # Si no son enteros, usar un selectbox
                valor = st.sidebar.selectbox(f"{col}", opciones[col])
        else:  # Columna numérica
            valor = st.sidebar.slider(f"{col}", opciones[col]["min"], opciones[col]["max"], opciones[col]["mean"], step=1)  # Usar slider con step=1

        datos_usuario[col] = valor
    else:
        st.warning(f"Columna '{col}' no encontrada en el dataset. Se usará un valor predeterminado.")
        datos_usuario[col] = 0  # O un valor predeterminado adecuado

# Crear un DataFrame con los datos del usuario
df_usuario = pd.DataFrame([datos_usuario])

# Realizar la predicción
if st.sidebar.button("Predecir"):
    try:
        prediccion = modelo.predict(df_usuario)  # Usar el DataFrame para la predicción
        resultado = "Positivo (1)" if prediccion[0] == 1 else "Negativo (0)"
        st.sidebar.success(f"*Predicción del modelo:* {resultado}")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")

# Botón para descargar los resultados
st.download_button(
                label="Descargar resultados",
                data=df_reporte.to_csv(index=False).encode('utf-8'),
                file_name="resultados_prediccion.csv",
                mime="text/csv"
            )   

# Link archivo de análisis
st.link_button("📥 Descargar Archivo desde GitHub", "https://github.com/MelRossi/App-Neuro-Tx-Latam/raw/main/An%C3%A1lisis_METS.ipynb")

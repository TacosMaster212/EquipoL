#Liberias utilizdas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import os

#Cambio de icono de la aplicación
st.set_page_config(page_title="Uniformidad de los hornos", page_icon="https://cdn.prod.website-files.com/67b62ae3d1b40c229065e285/67c88f9117d5c754a3bd28e0_NUTEC-Group_Logo.svg", layout="wide")

#Titulo de la aplicación
st.markdown("""
    <div style='display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    padding: 20px 40px;
    background: linear-gradient(to right, rgba(0,0,0,0.6), rgba(0,0,0,0.2)),  
            url("https://cdn.prod.website-files.com/67b62ae3d1b40c229065e285/67c4be82ec9198f05ae7ab68_ZxbAZYF3NbkBXyTF_Heat-transfer-program-bg-inverted-p-2000.jpg");
    background-size: cover;
    background-position: center;
    border-radius: 12px;
    gap: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: all 0.3s ease-in-out;'>
        <img src='https://cdn.prod.website-files.com/67b62ae3d1b40c229065e285/67c89626183dbac06e2629c3_NUTEC_Logo-Group.svg' style='height: 80px; margin-left: 30px; margin-right: 20px; border-radius: 7px;'>
        <div style='flex: 1; min-width: 300px; display: flex; justify-content: center;'>
        <h1 style='
            color: white;
            text-align: center;
            font-size: 32.8px;
            background-color: rgba(0,0,0,0.5);
            padding: 24px 24px;
            border-radius: 7px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.4);
            margin: 0;
            transition: all 0.3s ease-in-out;
        '>
            Análisis de Uniformidad Térmica en Hornos
        </h1>
    </div>
        <img src='https://tec.mx/sites/default/files/repositorio/Logotipo-Horizontal-black.svg' background-color:  style='background-color: rgba(255,255,255,1); height: 80px; margin-left: 30px; margin-right: 20px;  border-radius: 7px;'>
    </div>
""", unsafe_allow_html=True)

#Aplicar estilo personalizado inspirado en la página de Nutec
st.markdown("""
    <style>
    .stApp {
      background:
      color: white;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
      color: white;
      font-weight: bold;
    }
    [data-testid="stSidebar"] {
      background-image: url("https://static.wixstatic.com/media/d34cc5_6396eedf64664d588f8e1e62035896af~mv2.png/v1/fill/w_1281,h_437,al_c/d34cc5_6396eedf64664d588f8e1e62035896af~mv2.png");
      background-size: 200px;
      background-repeat: no-repeat;
      background-position: 30px 20px;
    }
    h3,
    label[data-testid="stMetricLabel"] p {
        font-weight: bold;
        color: #2D2E78;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5em;
        font-weight: bold;
        color: #82858d;
    }
    .stButton>button {
      background-color: #e50914;
      color: white;
      font-weight: bold;
      border-radius: 8px;
      padding: 0.5em 1.2em;
      border: none;
    }
    .stButton>button:hover {
      background-color: #b90710;
    }
    </style>
""", unsafe_allow_html=True)

# Eliminar basura streamlit
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Cargar los datos
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        # Ruta del archivo en el mismo directorio que Uni.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'Datos_para_entrenamiento.csv')
        return pd.read_csv(file_path)

df = load_data()

df[['Uniformidad Inf','Overshoot','Heat Up Time (min)','Lag Time (min)']] = df[['Uniformidad Inf','Overshoot','Heat Up Time (min)','Lag Time (min)']].astype('category')

#apartado del analisis de correlaciones por medio de Chi-cuadrada
st.header("Análisis de Chi-cuadrada")
possible_cat_vars = [
    'Gas o eléctrico',
    'Empty or loaded',
    'Tipo de horno',
    'Tipo de calentamiento [Entrada Escalón, Rampa o Receta]',
    'Tipo de aislamiento de las paredes (sin considerar el poyo)',
    'Tipo de control de presión',
    'Posición del termopar de control con respecto a la carga u horno',
    'Posición del termopar de control con respecto al bafle',
    'Posición del termopar en la carga ',
    'Tipo de probeta',
    'Tipo de Termopar de control',
    'Tipo de control de presión  [abiertos, contra pesados, lazo abierto, lazo cerrado]',
    'Dispositivo para control de Presión (VFD en extractor, Dampers de corderita, Flautas de aire).',
    'Tipo de control',
    'Cambios de aire ',
    'Que pueda arrancar el recirculador al 100% desde Frio',
    'Tipo de Recirculador (centrifugo, axial, con Scroll), tipo de scroll y tipo de aspas ',
    'Patrón de flujo del aire (Horizontal, vertica down directo, vertcical down con bafle, vertical up, directo o vertical up con bafle). ',
    'Reversibilidad del flujo de aire',
    'Tipo de bafle [tachas, nada, rejillas ajustables, circulos fijos]',
    'Baffle para ajustar el retorno ',
    'Venas en esquinas',
    'Venas ladronas de aire para rejillas',
    'Venas direccionadoras',
    'Posicion de quemador (Antes o despues del recirculador,  )',
    'Con Rompe flamas',
    'Tipo de tubo a la salida del quemador',
    'Quemador de tubo radiante',
    'Posicion de la toma del  Extractor ',
    'Posición de diluciones',
    'Regulador Cross connected',
    'Uso de Piers',
    'Con charola',
    'Posición de las extracciones de gases ',
    'Posición de quemadores',
    'Modo de control',
    'Exceso de aire Variable',
    'Flame less'
]
cat_vars = [var for var in possible_cat_vars if var in df.columns]

if st.checkbox("Mostrar datos crudos"):
    st.write(df.head())

# --- Sección para Uniformidad Suoperior ---
st.header("Uniformidad Sup")
variable_objetivo = "Uniformidad Sup"
# Lista para almacenar los resultados de la Chi-cuadrada
chi2_results = []

# Iterar sobre todas las variables categóricas y realizar la prueba
for var_select in cat_vars:
    if var_select and variable_objetivo in df.columns:
        contingency = pd.crosstab(df[var_select], df[variable_objetivo])
        # Asegurarse de que la tabla de contingencia tiene suficientes dimensiones
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p, dof, expected = chi2_contingency(contingency)
            chi2_results.append({'Variable': var_select, 'p-valor': p, 'Chi2': chi2})

# Convertir los resultados a un DataFrame para facilitar el manejo
chi22_df = pd.DataFrame(chi2_results)

# Ordenar por p-valor y seleccionar el top 5
if not chi22_df.empty:
    top_5_chi2 = chi22_df.sort_values(by='p-valor').head(5).reset_index(drop=True)

    # Sumar 1 al índice para que empiece desde 1
    top_5_chi2.index = top_5_chi2.index + 1


    st.subheader(f"Top 5 Variables con Mayor Relación con '{variable_objetivo}'")
    st.dataframe(top_5_chi2)

    # Selectbox para el análisis detallado
    if st.checkbox("Mostrar análisis detallado para Uniformidad Sup"):
        var_select_individual_velocidad = st.selectbox(
            f"Selecciona una variable categórica para ver el análisis detallado con {variable_objetivo}:",
            cat_vars,
            key="tus_selectbox_individual_uniformidad_sup"
        )

        if var_select_individual_velocidad and variable_objetivo in df.columns:
            st.subheader(f"Análisis detallado: {var_select_individual_velocidad} vs {variable_objetivo}")
            contingency = pd.crosstab(df[var_select_individual_velocidad], df[variable_objetivo])
            st.write("Tabla de contingencia:", contingency)
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(contingency)
                st.markdown(f"**Chi2:** {chi2:.2f}  ")
                st.markdown(f"**p-valor:** {p:.4f}  ")
                if p < 0.05:
                    st.success("Existe una relación estadísticamente significativa entre las variables.")
                else:
                    st.info("No se encontró una relación estadísticamente significativa.")
            else:
                st.warning("No hay suficientes categorías para realizar la prueba de Chi-cuadrada.")

else:
    st.info(f"No se pudieron realizar pruebas de Chi-cuadrada para ninguna variable categórica válida con {variable_objetivo}.")

# --- Sección para Velocidad del aire ---
st.header("Velocidad de aire")
variable_objetivo2 = 'Velocidad del aire'
# Lista para almacenar los resultados de la Chi-cuadrada
chi22_results = []

# Iterar sobre todas las variables categóricas y realizar la prueba
for var_select in cat_vars:
    if var_select and variable_objetivo2 in df.columns:
        contingency = pd.crosstab(df[var_select], df[variable_objetivo2])
        # Asegurarse de que la tabla de contingencia tiene suficientes dimensiones
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p, dof, expected = chi2_contingency(contingency)
            chi22_results.append({'Variable': var_select, 'p-valor': p, 'Chi2': chi2})

# Convertir los resultados a un DataFrame para facilitar el manejo
chi22_df = pd.DataFrame(chi22_results)

# Ordenar por p-valor y seleccionar el top 5
if not chi22_df.empty:
    top_5_chi22 = chi22_df.sort_values(by='p-valor').head(5).reset_index(drop=True)

    # Sumar 1 al índice para que empiece desde 1
    top_5_chi22.index = top_5_chi22.index + 1

    st.subheader(f"Top 5 Variables con Mayor Relación con '{variable_objetivo2}'")
    st.dataframe(top_5_chi22)

    # Selectbox para el análisis detallado
    if st.checkbox("Mostrar análisis detallado por variable para Velocidad del aire"):
        var_select_individual_velocidad = st.selectbox(
            f"Selecciona una variable categórica para ver el análisis detallado con {variable_objetivo2}:",
            cat_vars,
            key="tus_selectbox_individual_velocidad"
        )

        if var_select_individual_velocidad and variable_objetivo2 in df.columns:
            st.subheader(f"Análisis detallado: {var_select_individual_velocidad} vs {variable_objetivo2}")
            contingency = pd.crosstab(df[var_select_individual_velocidad], df[variable_objetivo2])
            st.write("Tabla de contingencia:", contingency)
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(contingency)
                st.markdown(f"**Chi2:** {chi2:.2f}  ")
                st.markdown(f"**p-valor:** {p:.4f}  ")
                if p < 0.05:
                    st.success("Existe una relación estadísticamente significativa entre las variables.")
                else:
                    st.info("No se encontró una relación estadísticamente significativa.")
            else:
                st.warning("No hay suficientes categorías para realizar la prueba de Chi-cuadrada.")

else:
    st.info(f"No se pudieron realizar pruebas de Chi-cuadrada para ninguna variable categórica válida con {variable_objetivo2}.")



if variable_objetivo in df.columns:
    # Variables que no deben participar en el modelo
    variables_a_excluir = ['Uniformidad Sup', 'Heat Up Time (min)', 'Lag Time (min)']
    df_ml = df.drop(columns=variables_a_excluir, errors='ignore').select_dtypes(include=['number']).copy()

    # Codificación con factorize (guarda mapeo de clases)
    df_ml[variable_objetivo], class_labels = pd.factorize(df[variable_objetivo])

    if df_ml.shape[1] > 1:
        X = df_ml.drop(columns=[variable_objetivo])
        y = df_ml[variable_objetivo]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Importancia de variables
        st.subheader("Importancia de variables:")
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances_sorted = importances.sort_values(ascending=False)

        # Invertir cada nombre de variable en la lista de índices
        inverted_labels = [label[::-1] for label in importances_sorted.index]

        fig = px.bar(
            x=importances_sorted.index,
            y=importances_sorted.values,
            labels={"x": "Variable", "y": "Importancia"},
            title="Importancia de variables ordenada de mayor a menor",
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig)

        top_features = importances.sort_values(ascending=False).head(8).index.tolist()
         
        #Inicio de la predicci+on Random Forest
        st.header("Predicción para un Horno Nuevo")
        st.write("Ingresa los valores solo para las variables más importantes. El resto se completará automáticamente con la mediana.")
        user_inputs_for_prediction = {}

        with st.form("prediction_form"):
            st.write("Por favor, introduce los valores para las variables clave del horno:")
            input_cols = st.columns(2)
            for i, feat_name in enumerate(top_features):
                col_idx = i % 2
                with input_cols[col_idx]:
                    min_val = float(df[feat_name].min()) if not pd.isna(df[feat_name].min()) else 0.0
                    max_val = float(df[feat_name].max()) if not pd.isna(df[feat_name].max()) else 100.0
                    default_val = float(df[feat_name].median()) if not pd.isna(df[feat_name].median()) else (min_val + max_val) / 2
                    user_inputs_for_prediction[feat_name] = st.number_input(
                        f"{feat_name} (Numérico)",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=(max_val - min_val) / 100 if (max_val - min_val) > 0 else 1.0,
                        format="%.2f",
                        key=f"pred_input_{feat_name}"
                    )
            submitted = st.form_submit_button("Obtener Predicción")

            if submitted:
                new_oven_data = {col: df[col].median() if col in df.columns else 0.0 for col in X.columns}
                for feat, val in user_inputs_for_prediction.items():
                    new_oven_data[feat] = val
                input_df_for_prediction = pd.DataFrame([new_oven_data])[X.columns]

                prediction = model.predict(input_df_for_prediction)[0]
                resultado = "Passed" if prediction == 1 else "Failed"
                ...
                st.subheader("Visualización del Comportamiento de la Uniformidad en el Tiempo")
                if 'SetPoint' in user_inputs_for_prediction and 'Operating Temperature' in user_inputs_for_prediction:
                    sp = user_inputs_for_prediction['SetPoint']
                    op_temp = user_inputs_for_prediction['Operating Temperature']
                    uniformidad = abs(sp - op_temp)

                    modo_simulacion = st.selectbox("Tipo de prueba térmica", ["Escalón", "Rampa"])
                    tiempo_total = st.slider("Duración de la simulación (min)", min_value=25, max_value=360, value=1, step=10)
                    intervalo_min = st.slider("Intervalo de muestreo (min)", min_value=1, max_value=10, value=1)
                    tiempo = np.arange(0, tiempo_total, intervalo_min)
                    temperatura_simulada = []

                    for t in tiempo:
                        if modo_simulacion == "Escalón":
                            if t < 20:
                                temperatura_simulada.append(op_temp)
                            elif t < 60:
                                step = (sp - op_temp) * ((t - 20) / 40)
                                temperatura_simulada.append(op_temp + step)
                            else:
                                fluctuacion = np.random.uniform(-uniformidad * 0.1, uniformidad * 0.1)
                                temperatura_simulada.append(sp + fluctuacion)
                        elif modo_simulacion == "Rampa":
                            if t <= tiempo_total:
                                temp = op_temp + ((sp - op_temp) * (t / tiempo_total))
                                fluctuacion = np.random.uniform(-uniformidad * 0.05, uniformidad * 0.05)
                                temperatura_simulada.append(temp + fluctuacion)

                    fig_linea = px.line(
                        x=tiempo,
                        y=temperatura_simulada,
                        labels={"x": "Tiempo (min)", "y": "Temperatura (°F)"},
                        title=f"Simulación tipo {modo_simulacion} de la Uniformidad en el Tiempo"
                    )
                    fig_linea.add_scatter(
                        x=tiempo,
                        y=[sp + uniformidad] * len(tiempo),
                        mode='lines',
                        name='SetPoint + Uniformidad',
                        line=dict(dash='dot', color='green')
                    )
                    fig_linea.add_scatter(
                        x=tiempo,
                        y=[sp - uniformidad] * len(tiempo),
                        mode='lines',
                        name='SetPoint - Uniformidad',
                        line=dict(dash='dot', color='red')
                    )
                    fig_linea.add_scatter(
                        x=tiempo,
                        y=[sp] * len(tiempo),
                        mode='lines',
                        name='SetPoint',
                        line=dict(dash='dash', color='black')
                    )
                    st.plotly_chart(fig_linea)
                else:
                    st.info("No se pudo generar la curva de comportamiento térmico. Asegúrate de ingresar 'SetPoint' y 'Operating Temperature'.")


                # Mostrar el resultado principal
                st.subheader("Resultado de la Clasificación")

                # Calcular uniformidad si SetPoint y Operating Temperature están presentes
                if 'SetPoint' in user_inputs_for_prediction and 'Operating Temperature' in user_inputs_for_prediction:
                    sp = user_inputs_for_prediction['SetPoint']
                    op_temp = user_inputs_for_prediction['Operating Temperature']
                    # Definir estado estable como los últimos 20 minutos o el 25% final del tiempo total
                    n_estable = 20 // intervalo_min  # cuantos puntos equivalen a 20 minutos
                    estado_estable = temperatura_simulada[-n_estable:]

                    # Uniformidad como el rango (máximo - mínimo) en estado estable, o desviación estándar si prefieres
                    uniformidad = np.mean([abs(t - sp) for t in estado_estable])  # También puedes usar: max - min

                    # Estimar lag time y heat up time con funciones simples
                    lag_time = np.clip(uniformidad * 0.8, 1, 60)  
                    heat_up_time = np.clip((sp - op_temp) * 0.5 + 10, 5, 120)  

                    st.subheader("🔍 Predicción de Métricas Térmicas del Horno")

                    def styled_metric(nombre, valor, unidad, color):
                        st.markdown(f"""
                            <div style="background-color:{color};padding:15px;border-radius:10px;margin-bottom:10px">
                                <h5 style="color:white;margin:0">{nombre}</h5>
                                <h2 style="color:white;margin:0">{valor:.2f} {unidad}</h2>
                            </div>
                        """, unsafe_allow_html=True)

                    styled_metric("Uniformidad estimada", uniformidad, "°F", "#1f77b4") 
                    styled_metric("Lag Time estimado", lag_time, "min", "#ff7f0e") 
                    styled_metric("Heat Up Time estimado", heat_up_time, "min", "#2ca02c")    



import streamlit as st
import pandas as pd
import pickle
import joblib
import os
import numpy as np
import plotly.express as px
from datetime import datetime
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import base64

# Función para convertir imagen a Base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Configuración de la página con tema oscuro e imagen de fondo
def set_background_image():
    # Obtener la ruta completa del archivo
    bg_image_path = os.path.join(os.path.dirname(__file__), "fondo.jpg")
    encoded_image = get_base64_of_bin_file(bg_image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(18, 18, 18, 0.85), rgba(18, 18, 18, 0.85)), 
                        url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #ffffff;
        }}
        
        /* Asegurar que los contenedores sean semi-transparentes */
        .main-container, .custom-metric, .stExpander, .stAlert, .plot-container {{
            background-color: rgba(30, 30, 30, 0.9) !important;
            backdrop-filter: blur(2px);
        }}
        
        /* Mejorar legibilidad de texto sobre fondo */
        h1, h2, h3, h4, h5, h6, p, div, span {{
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Llamar a la función para establecer la imagen de fondo
set_background_image()

# Resto de tu configuración CSS original
st.markdown("""
<style>
/* Contenedores principales */
.main-container {
    background-color: rgba(30, 30, 30, 0.9);
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
    border: 1px solid #333;
}

/* Títulos y encabezados */
h1, h2, h3, h4 {
    color: #4fc3f7;
    font-weight: 600;
}

/* Texto general */
body {
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Tarjetas de métricas */
.custom-metric {
    border-left: 4px solid #4fc3f7;
    padding: 15px;
    background-color: rgba(37, 37, 37, 0.9);
    border-radius: 8px;
    margin-bottom: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}

/* Botones */
.stButton>button {
    border-radius: 8px;
    padding: 10px 20px;
    background-color: #1976d2;
    color: white;
    font-weight: 500;
    border: none;
    transition: all 0.3s;
}



.stButton>button:hover {
    background-color: #1565c0;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}

/* Pestañas */
.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 8px 8px 0 0;
    background-color: rgba(51, 51, 51, 0.9);
    transition: all 0.3s;
    color: #b0b0b0;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(30, 30, 30, 0.9);
    font-weight: 600;
    color: #4fc3f7;
    border-bottom: 2px solid #4fc3f7;
}

/* Expanders */
.stExpander {
    background-color: rgba(37, 37, 37, 0.9);
    border-radius: 8px;
    border: 1px solid #333;
}

.stExpander .stExpanderHeader {
    color: #4fc3f7;
}

/* Inputs */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
    background-color: rgba(37, 37, 37, 0.9);
    color: white;
    border: 1px solid #333;
}

.stSlider .st-ae {
    color: #4fc3f7;
}

/* Tooltips */
[data-testid="stTooltipContent"] {
    font-size: 14px;
    background-color: #333;
    color: white;
    border: 1px solid #444;
}

.footer {
    color: #9e9e9e;
    font-size: 0.9rem;
    text-align: center;
    padding: 20px;
    margin-top: 40px;
    border-top: 1px solid #333;
    background-color: rgba(25, 25, 25, 0.8);
    backdrop-filter: blur(5px);
    border-radius: 0 0 10px 10px;
}

.footer a:hover {
    color: #81d4fa;
    text-decoration: underline;
}

/* Efecto de separación del contenido principal */
.stApp {
    padding-bottom: 60px;
}

/* Alertas */
.stAlert {
    border-radius: 8px;
    background-color: rgba(37, 37, 37, 0.9);
    border: 1px solid #333;
}

/* Gráficos */
.plot-container {
    background-color: rgba(30, 30, 30, 0.9);
    border-radius: 8px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# Ruta base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar dataset para inputs con caché
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "thyroid_cancer_risk_data.csv"))

# --- Encabezado con tema oscuro ---
st.markdown("""
<div style="background: linear-gradient(to right, #0d47a1, #1976d2); padding: 25px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
    <h1 style="color: white; margin: 0; text-shadow: 1px 1px 3px rgba(0,0,0,0.3);">ThyroScan </h1>
    <p style="color: #e3f2fd; margin: 0; font-size: 1.1rem;">Sistema de evaluación de nódulos tiroideos</p>
</div>
""", unsafe_allow_html=True)

# División en pestañas
tab1, tab2, tab3 = st.tabs(["📋 Formulario de Evaluación", "📊 Exploración de Datos", "ℹ️ Acerca del Modelo"])

with tab1:
    data = load_data()
    
    # Layout de entradas mejorado
    with st.container():
        st.subheader("📋 Información del Paciente")
        col1, col2 = st.columns(2)
        
        
        with col1:
            patient_name = st.text_input("Nombre del Paciente", 
                                      placeholder="Ingrese el nombre completo del paciente",
                                      help="Nombre completo del paciente para identificación")
                
            gender = st.selectbox("Género", sorted(data["Gender"].dropna().unique()), 
                                help="Seleccione el género del paciente")
            age = st.slider("Edad", 0, 100, 45, 
                           help="Edad del paciente en años")
            country = st.selectbox("País", sorted(data["Country"].dropna().unique()),
                                 help="País de origen del paciente")
            
        with col2:

            ethnicity = st.selectbox("Etnicidad", sorted(data["Ethnicity"].dropna().unique()),
                                   help="Grupo étnico del paciente")
            family_history = st.radio("Antecedentes familiares de cáncer tiroideo", 
                                    sorted(data["Family_History"].dropna().unique()),
                                    horizontal=True)
            radiation = st.radio("Exposición a radiación ionizante", 
                               sorted(data["Radiation_Exposure"].dropna().unique()),
                               horizontal=True)
    
    # Sección clínica
    with st.expander("🏥 Datos Clínicos y Bioquímicos", expanded=True):
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Perfil Tiroideo**")
            tsh = st.number_input("TSH (mIU/L)", min_value=0.0, max_value=100.0, value=2.5, step=0.1,
                                help="Nivel de hormona estimulante de la tiroides")
            t3 = st.number_input("T3 libre (pg/mL)", min_value=0.0, max_value=20.0, value=3.5, step=0.1)
            t4 = st.number_input("T4 libre (ng/dL)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
            
        with col4:
            st.markdown("**Características del Nódulo**")
            nodule_size = st.number_input("Tamaño del nódulo (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1,
                                         help="Diámetro mayor del nódulo en centímetros")
    
    # Factores de riesgo adicionales
    with st.expander("⚠️ Factores de Riesgo Adicionales", expanded=False):
        col5, col6 = st.columns(2)
        
        with col5:
            diabetes = st.radio("Diabetes", sorted(data["Diabetes"].dropna().unique()),
                              horizontal=True)
            obesity = st.radio("Obesidad (IMC ≥30)", sorted(data["Obesity"].dropna().unique()),
                             horizontal=True)
            
        with col6:
            iodine = st.radio("Deficiencia de yodo", sorted(data["Iodine_Deficiency"].dropna().unique()),
                              horizontal=True,help="Estado nutricional de yodo")
            smoking = st.radio("Hábito tabáquico", sorted(data["Smoking"].dropna().unique()),
                             horizontal=True,help="Consumo actual o previo de tabaco")
    
    # Cargar modelo, encoders y scaler con caché
    @st.cache_resource
    def load_model_components():
        try:
            encoders = {}
            nominal_cols = ['Gender', 'Country', 'Ethnicity', 'Family_History', 'Radiation_Exposure',
                          'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']
            
            for col in nominal_cols:
                path = os.path.join(BASE_DIR, f"encoders/label_encoder_{col}.pkl")
                if os.path.exists(path):
                    encoders[col] = joblib.load(path)
                else:
                    st.error(f"Encoder faltante: {col}")
                    return None, None, None, None
            
            scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
            model = joblib.load(os.path.join(BASE_DIR, "xgb_diagnosis_model_smote.pkl"))
            
            with open(os.path.join(BASE_DIR, "model_columns.pkl"), "rb") as f:
                model_columns = pickle.load(f)
                
            return encoders, scaler, model, model_columns
        
        except Exception as e:
            st.error(f"Error al cargar componentes del modelo: {str(e)}")
            return None, None, None, None

    # Botón de predicción
    if st.button("🔍 Evaluar Riesgo", type="primary", use_container_width=True):
        with st.spinner("Analizando datos..."):
            encoders, scaler, model, model_columns = load_model_components()
            
            if None in [encoders, scaler, model, model_columns]:
                st.error("No se pudo cargar el modelo. Por favor verifique los archivos necesarios.")
                st.stop()
            
            # Preparar entrada del usuario
            entrada = pd.DataFrame({
                'Patient_Name': [patient_name],
                'Gender': [gender],
                'Country': [country],
                'Ethnicity': [ethnicity],
                'Family_History': [family_history],
                'Radiation_Exposure': [radiation],
                'Iodine_Deficiency': [iodine],
                'Smoking': [smoking],
                'Obesity': [obesity],
                'Diabetes': [diabetes],
                'Age': [age],
                'TSH_Level': [tsh],
                'T3_Level': [t3],
                'T4_Level': [t4],
                'Nodule_Size': [nodule_size]
            })
            
            # Validación de datos de entrada
            if nodule_size <= 0:
                st.warning("El tamaño del nódulo debe ser mayor que cero")
                st.stop()
            
            if age < 18 and family_history == 'Yes':
                st.warning("⚠️ Paciente pediátrico con antecedentes familiares - considerar evaluación especializada")
            
            # Codificación y escalado
            try:
                for col in encoders:
                    entrada[col] = encoders[col].transform(entrada[col])
                
                num_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
                entrada[num_cols] = scaler.transform(entrada[num_cols])
                entrada = entrada[model_columns]
                
                # Predicción
                y_proba = model.predict_proba(entrada)[0]
                y_pred = (y_proba[1] >= 0.44).astype(int)  # Usando el umbral óptimo
                
                # Mostrar resultados
                st.success("✅ Evaluación completada")
                
                # Tarjetas de resultados
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.markdown(f"""
                    <div class="custom-metric">
                        <h3 style="margin: 0; color: #81c784;">Probabilidad de benignidad</h3>
                        <p style="font-size: 1.5rem; margin: 0; color: #e0e0e0;">{y_proba[0]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f"""
                    <div class="custom-metric">
                        <h3 style="margin: 0; color: #ff8a65;">Probabilidad de malignidad</h3>
                        <p style="font-size: 1.5rem; margin: 0; color: #e0e0e0;">{y_proba[1]*100:.1f}%</p>
                        <p style="margin: 0; font-size: 0.8rem; color: #b0b0b0;">Umbral diagnóstico: 44%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res3:
                    risk_level = "Alto Riesgo" if y_pred == 1 else "Bajo Riesgo"
                    color = "#ff8a65" if y_pred == 1 else "#81c784"
                    st.markdown(f"""
                    <div style="border-left: 5px solid {color}; padding: 0.5rem; background-color: #252525; border-radius: 0.5rem;">
                        <h3 style="color: {color}; margin: 0;">{risk_level}</h3>
                        <p style="margin: 0; color: #e0e0e0;">{'Se recomienda evaluación especializada' if y_pred == 1 else 'Seguimiento rutinario recomendado'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualización interactiva con Plotly
                fig = px.pie(
                    names=['Benigno', 'Maligno'],
                    values=y_proba,
                    title='Distribución de Probabilidades',
                    hole=0.4,
                    color=['Benigno', 'Maligno'],
                    color_discrete_map={'Benigno':'#81c784', 'Maligno':'#ff8a65'}
                )
                fig.update_layout(
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font_color='#e0e0e0',
                    legend_font_color='#e0e0e0'
                )
                fig.update_traces(textinfo='percent+label', pull=[0, 0.1] if y_pred == 1 else [0, 0])
                st.plotly_chart(fig, use_container_width=True)
                
                # Recomendaciones basadas en el riesgo
                if y_pred == 1:
                    st.error("""
                    **🔴 Resultado sugestivo de malignidad**  
                    Recomendaciones:
                    - Consulta urgente con endocrinólogo
                    - Ecografía tiroidea con punción aspirativa con aguja fina (PAAF)
                    - Evaluación de ganglios linfáticos cervicales
                    - Considerar perfil tiroideo completo
                    """)
                else:
                    st.success("""
                    **🟢 Resultado sugestivo de benignidad**  
                    Recomendaciones:
                    - Seguimiento ecográfico en 6-12 meses
                    - Repetir perfil tiroideo en 3 meses si hay alteraciones
                    - Control clínico anual
                    """)
                
                # Descarga de informe
                report = f"""
                **Informe de Evaluación ThyroScan AI**  
                Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
                
                **Resultados:**  
                - Probabilidad de benignidad: {y_proba[0]*100:.1f}%  
                - Probabilidad de malignidad: {y_proba[1]*100:.1f}%  
                - Clasificación: {'Maligno (alto riesgo)' if y_pred == 1 else 'Benigno (bajo riesgo)'}  
                
                **Datos del Paciente:**  
                - Edad: {age} años  
                - Género: {gender}  
                - Tamaño del nódulo: {nodule_size} cm  
                - TSH: {tsh} mIU/L  
                
                **Recomendaciones:**  
                {'Consulta especializada recomendada' if y_pred == 1 else 'Seguimiento rutinario recomendado'}
                """

                st.download_button(
                    label="📄 Descargar Informe",
                    data=report,
                    file_name=f"thyroscan_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:

                st.error(f"Error en el procesamiento: {str(e)}")

    if st.button("📊 Cargar Datos a Google Sheets", 
             type="primary", 
             use_container_width=True,
             help="Guardar los datos del paciente y resultados en Google Sheets",
             key="google_sheets_button"):
        try:
            # Primero ejecutamos la predicción para obtener y_proba
            with st.spinner("Realizando predicción antes de guardar..."):
                encoders, scaler, model, model_columns = load_model_components()
                
                if None in [encoders, scaler, model, model_columns]:
                    st.error("No se pudo cargar el modelo. Verifique los archivos necesarios.")
                    st.stop()
                
                # Preparar entrada para el modelo
                entrada_modelo = pd.DataFrame({
                    'Gender': [gender],
                    'Country': [country],
                    'Ethnicity': [ethnicity],
                    'Family_History': [family_history],
                    'Radiation_Exposure': [radiation],
                    'Iodine_Deficiency': [iodine],
                    'Smoking': [smoking],
                    'Obesity': [obesity],
                    'Diabetes': [diabetes],
                    'Age': [age],
                    'TSH_Level': [tsh],
                    'T3_Level': [t3],
                    'T4_Level': [t4],
                    'Nodule_Size': [nodule_size]
                })
                
                # Codificación y escalado
                for col in encoders:
                    entrada_modelo[col] = encoders[col].transform(entrada_modelo[col])
                
                num_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
                entrada_modelo[num_cols] = scaler.transform(entrada_modelo[num_cols])
                entrada_modelo = entrada_modelo[model_columns]
                
                # Predicción
                y_proba = model.predict_proba(entrada_modelo)[0]
                prob_benigno = round(y_proba[0]*100, 2)  # Probabilidad de benignidad en %
                prob_maligno = round(y_proba[1]*100, 2)  # Probabilidad de malignidad en %

            # Conectar a Google Sheets
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds_path = os.path.join(os.path.dirname(__file__), "credenciales.json")
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            client = gspread.authorize(creds)
            sheet = client.open("Registro de usuarios").sheet1
            
            # Obtener fecha y hora actual
            now = datetime.now()
            fecha_actual = now.strftime("%Y-%m-%d")
            hora_actual = now.strftime("%H:%M:%S")
            
            # Preparar datos para Google Sheets
            entrada_gsheets = pd.DataFrame({
                'Fecha': [fecha_actual],
                'Hora': [hora_actual],
                'Patient_Name': [str(patient_name)],
                'Gender': [str(gender)],
                'Country': [str(country)],
                'Ethnicity': [str(ethnicity)],
                'Family_History': [str(family_history)],
                'Radiation_Exposure': [str(radiation)],
                'Iodine_Deficiency': [str(iodine)],
                'Smoking': [str(smoking)],
                'Obesity': [str(obesity)],
                'Diabetes': [str(diabetes)],
                'Age': [int(age)],
                'TSH_Level': [float(tsh)],
                'T3_Level': [float(t3)],
                'T4_Level': [float(t4)],
                'Nodule_Size': [float(nodule_size)],
                'Prob_Benigno (%)': [prob_benigno],  
                'Prob_Maligno (%)': [prob_maligno]
            })
        
            # Convertir y enviar a Google Sheets
            fila = []
            for value in entrada_gsheets.iloc[0]:
                if pd.api.types.is_integer(value):
                    fila.append(int(value))
                elif pd.api.types.is_float(value):
                    fila.append(float(value))
                else:
                    fila.append(str(value))
            
            sheet.append_row(fila)
            
            # Formatear las celdas de porcentaje (opcional)
            last_row = len(sheet.get_all_records()) 
            sheet.format(f"R{last_row+1}:S{last_row+1}", {
                "numberFormat": {
                    "type": "NUMBER",
                    "pattern": "0.00\"%\""
                }
            })
            
            st.success(f"✅ Datos y predicciones guardados exitosamente el {fecha_actual}")
        
        except Exception as e:
            st.error(f"❌ Error al procesar y guardar datos: {str(e)}")

with tab2:
    st.header("📊 Exploración de Datos Clínicos")
    data = load_data()
    
    # Filtros interactivos
    with st.expander("🔍 Filtros", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            age_range = st.slider(
                "Rango de edad",
                min_value=int(data['Age'].min()),
                max_value=int(data['Age'].max()),
                value=(20, 70)
            )
            diagnosis_filter = st.multiselect(
                "Diagnóstico",
                options=data['Diagnosis'].unique(),
                default=data['Diagnosis'].unique()
            )
            
        with col2:
            tsh_range = st.slider(
                "Rango de TSH",
                min_value=float(data['TSH_Level'].min()),
                max_value=float(data['TSH_Level'].max()),
                value=(0.5, 5.0)
            )
            size_range = st.slider(
                "Tamaño del nódulo",
                min_value=float(data['Nodule_Size'].min()),
                max_value=float(data['Nodule_Size'].max()),
                value=(0.5, 3.0)
            )
    
    # Aplicar filtros
    filtered_data = data[
        (data['Age'].between(age_range[0], age_range[1])) &
        (data['Diagnosis'].isin(diagnosis_filter)) &
        (data['TSH_Level'].between(tsh_range[0], tsh_range[1])) &
        (data['Nodule_Size'].between(size_range[0], size_range[1]))
    ]
    
    # Mostrar estadísticas 
    st.subheader("📈 Estadísticas Descriptivas")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Pacientes en dataset", len(filtered_data))
    
    with col4:
        benign_count = len(filtered_data[filtered_data['Diagnosis'] == 'Benign'])
        st.metric("Casos benignos", f"{benign_count} ({benign_count/len(filtered_data)*100:.1f}%)")
    
    with col5:
        malignant_count = len(filtered_data[filtered_data['Diagnosis'] == 'Malignant'])
        st.metric("Casos malignos", f"{malignant_count} ({malignant_count/len(filtered_data)*100:.1f}%)")

    # Gráfico 1: Distribución de diagnósticos por tamaño de nódulo (Histograma)
    st.subheader("Distribución de Tamaños de Nódulo por Diagnóstico")
    
    fig_size_dist = px.histogram(filtered_data,
                               x='Nodule_Size',
                               color='Diagnosis',
                               nbins=20,
                               barmode='overlay',
                               opacity=0.7,
                               color_discrete_map={'Benign': '#81c784', 'Malignant': '#ff8a65'},
                               title='Distribución de tamaños de nódulo',
                               labels={'Nodule_Size': 'Tamaño del nódulo (cm)', 'count': 'Número de casos'})
    
    fig_size_dist.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='#e0e0e0',
        legend_font_color='#e0e0e0',
        xaxis_title='Tamaño del nódulo (cm)',
        yaxis_title='Número de casos'
    )
    
    # Añadir línea vertical para tamaño umbral clínico (1 cm)
    fig_size_dist.add_vline(x=1.0, line_dash="dash", line_color="yellow", 
                          annotation_text="Umbral clínico 1cm", 
                          annotation_position="top")
    
    st.plotly_chart(fig_size_dist, use_container_width=True)
    
    # Gráfico 2: Porcentaje de malignidad por grupos de edad
    st.subheader("Riesgo de Malignidad por Grupos de Edad")
    
    # Crear grupos de edad
    age_bins = [0, 20, 30, 40, 50, 60, 70, 100]
    age_labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    filtered_data['Age_Group'] = pd.cut(filtered_data['Age'], bins=age_bins, labels=age_labels)
    
    # Calcular porcentaje de malignidad por grupo
    malignancy_rate = filtered_data.groupby('Age_Group')['Diagnosis'].apply(
        lambda x: (x == 'Malignant').mean() * 100).reset_index()
    
    fig_age_risk = px.bar(malignancy_rate,
                        x='Age_Group',
                        y='Diagnosis',
                        color='Diagnosis',
                        color_continuous_scale='OrRd',
                        title='Porcentaje de malignidad por grupo de edad',
                        labels={'Diagnosis': '% de Malignidad', 'Age_Group': 'Grupo de edad'})
    
    fig_age_risk.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='#e0e0e0',
        showlegend=False,
        xaxis_title='Grupo de edad',
        yaxis_title='Porcentaje de malignidad (%)'
    )
    st.plotly_chart(fig_age_risk, use_container_width=True)
    # Mapa coroplético
    st.subheader("🌍 Distribución de Pacientes por País (Mapa Coroplético)")
    
    # Códigos ISO Alpha-3 para Plotly 
    country_codes = {
        'Brazil': 'BRA',
        'China': 'CHN',
        'Germany': 'DEU',
        'India': 'IND',
        'Japan': 'JPN',
        'Nigeria': 'NGA',
        'Russia': 'RUS',
        'South Korea': 'KOR',
        'UK': 'GBR',
        'USA': 'USA'
    }
    
    # Procesar datos
    country_data = filtered_data['Country'].value_counts().reset_index()
    country_data.columns = ['Country', 'Patient_Count']
    country_data['ISO_Code'] = country_data['Country'].map(country_codes)
    
    # Crear mapa coroplético
    fig_choropleth = px.choropleth(country_data,
                                  locations="ISO_Code",
                                  color="Patient_Count",
                                  hover_name="Country",
                                  hover_data=["Patient_Count"],
                                  projection="natural earth",
                                  color_continuous_scale="teal",
                                  title="Distribución de pacientes por país",
                                  labels={'Patient_Count': 'Número de pacientes'})
    
    # Personalizar estilo para tema oscuro
    fig_choropleth.update_geos(
        bgcolor='#1e1e1e',
        showcountries=True,
        countrycolor='#444444',
        showocean=True,
        oceancolor='#121212',
        showframe=False
    )
    
    fig_choropleth.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='#e0e0e0',
        margin={"r":0,"t":40,"l":0,"b":0},
        height=500,
        coloraxis_colorbar=dict(
            title="N° Pacientes",
            thickness=15,
            len=0.6
        )
    )
    
    st.plotly_chart(fig_choropleth, use_container_width=True)
    
    # Gráfico complementario de tipo treemap
    st.subheader("📊 Proporción por Países")
    
    fig_treemap = px.treemap(country_data,
                            path=['Country'],
                            values='Patient_Count',
                            color='Patient_Count',
                            color_continuous_scale='blues',
                            hover_data=['Patient_Count'],
                            title='Proporción de pacientes por país')
    
    fig_treemap.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='#e0e0e0',
        margin={"t":40,"l":0,"r":0,"b":0}
    )
    
    st.plotly_chart(fig_treemap, use_container_width=True)

    st.subheader("🔥 Prevalencia de Factores de Riesgo por País")

    # Preparar datos (pivot table)
    risk_by_country = pd.crosstab(filtered_data['Ethnicity'], filtered_data['Diagnosis'])
    risk_by_country['Malignancy_Rate'] = (risk_by_country['Malignant'] / (risk_by_country['Malignant'] + risk_by_country['Benign'])) * 100

    fig_heatmap = px.imshow(risk_by_country[['Malignancy_Rate']].T,
                        color_continuous_scale='orrd',
                        labels=dict(x="País", y="", color="Tasa %"),
                        title='Tasa de malignidad por país')

    fig_heatmap.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='#e0e0e0'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


    
    

with tab3:
    st.header("ℹ️ Acerca del Modelo ThyroScan")
    
    st.markdown("""
    ### Modelo de Predicción de Riesgo de Cáncer Tiroideo
    
    Este sistema utiliza un modelo de aprendizaje automático XGBoost optimizado para evaluar 
    el riesgo de malignidad en nódulos tiroideos basado en características clínicas y demográficas.
    """)
    
    with st.expander("📊 Métricas de Rendimiento", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Evaluación en Conjunto de Prueba:**
            - **Exactitud (Accuracy):** 78.56%
            - **Precisión (Maligno):** 84%
            - **Sensibilidad (Recall):** 71%
            - **F2-Score:** 72.91%
            """)
            
        with col2:
            st.markdown("""
            **Matriz de Confusión:**
            |                 | Predicción Benigna | Predicción Maligna |
            |-----------------|--------------------|--------------------|
            | **Real Benigna** | 28,249             | 4,391              |
            | **Real Maligna** | 9,602              | 23,037             |
            """)
    
    with st.expander("🛠️ Detalles Técnicos", expanded=False):
        st.markdown("""
        **Arquitectura del Modelo:**
        - Algoritmo: XGBoost (Gradient Boosting)
        - Balanceo de datos: SMOTE (Synthetic Minority Over-sampling Technique)
        - Hiperparámetros optimizados mediante búsqueda bayesiana
        
        **Variables Incluidas:**
        1. Datos demográficos (edad, género, etnicidad)
        2. Antecedentes clínicos (familiares, radiación)
        3. Perfil tiroideo (TSH, T3, T4)
        4. Características del nódulo (tamaño)
        5. Factores de riesgo (tabaquismo, obesidad, diabetes)
        """)
    
    with st.expander("📝 Limitaciones y Consideraciones", expanded=False):
        st.markdown("""
        - El modelo tiene un rendimiento óptimo para nódulos entre 1-4 cm
        - La precisión disminuye en pacientes menores de 18 años
        - No considera hallazgos ecográficos (microcalcificaciones, vascularización)
        - No reemplaza el juicio clínico especializado
        
        **Uso recomendado:**  
        Herramienta de apoyo para la toma de decisiones, especialmente en entornos 
        con recursos limitados o para priorización de casos.
        """)
    
    st.markdown("""
    ### 🔍 Interpretación de Resultados
    
    - **Probabilidad <30%:** Muy baja probabilidad de malignidad
    - **Probabilidad 30-44%:** Baja probabilidad, seguimiento recomendado
    - **Probabilidad 44-70%:** Riesgo intermedio, requiere evaluación adicional
    - **Probabilidad >70%:** Alta probabilidad de malignidad
    """)

# Footer mejorado con derechos reservados
st.markdown("""
<div class="footer">
    <div style="margin-bottom: 5px;">
        ThyroScan AI v1.0 · © 2025 Thyroid Diagnostics Inc.
    </div>
    <div style="margin-bottom: 5px; font-size: 0.85rem;">
        Todos los derechos reservados a los creadores: Maria Camila Castillo, Juan David Plata, Laura Daniela Sierra y Juan Francisco Rojas
    </div>
    <div style="margin-top: 10px;">
        <a href="#" style="color: #4fc3f7; text-decoration: none; margin-right: 15px;">Términos de uso</a> 
        <a href="#" style="color: #4fc3f7; text-decoration: none; margin-right: 15px;">Política de privacidad</a>
        <a href="#" style="color: #4fc3f7; text-decoration: none;">Contacto</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Advertencia final 
st.warning("""
**Nota importante:** Esta herramienta es un sistema de apoyo a la decisión clínica y no sustituye 
el juicio profesional. Los resultados deben interpretarse en el contexto clínico completo del paciente.
""")





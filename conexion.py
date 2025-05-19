import gspread

from oauth2client.service_account import ServiceAccountCredentials
 
# Definir el alcance

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
 
# Cargar las credenciales

creds = ServiceAccountCredentials.from_json_keyfile_name('credenciales.json', scope)
 
# Autenticarse con Google

try:

    client = gspread.authorize(creds)

    print("Conexión exitosa a Google Sheets")

except Exception as e:

    print(f"Error de autenticación: {e}")
 
# Probar acceso a la hoja de cálculo

try:

    sheet = client.open("Registro de usuarios").sheet1

    data = sheet.get_all_records()  # Obtén todas las filas como diccionarios

    print(data)  # Muestra los datos obtenidos

except Exception as e:

    print(f"Error al acceder a la hoja de cálculo: {e}")



 if st.button("Cargar Datos a Google Sheets"):
    try:
        # Primero ejecutamos la predicción para obtener y_proba
        with st.spinner("Realizando predicción antes de guardar..."):
            encoders, scaler, model, model_columns = load_model_components()
            
            if None in [encoders, scaler, model, model_columns]:
                st.error("No se pudo cargar el modelo. Verifique los archivos necesarios.")
                st.stop()
            
            # Preparar entrada para el modelo (código existente)
            # ... [tu código de predicción aquí] ...
            
            y_proba = model.predict_proba(entrada_modelo)[0]
            prob_benigno = round(y_proba[0]*100, 2)
            prob_maligno = round(y_proba[1]*100, 2)

        # Conectar a Google Sheets
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credenciales.json", scope)
        client = gspread.authorize(creds)
        
        # Abrir la hoja con manejo de headers duplicados
        try:
            sheet = client.open("Registro de usuarios").sheet1
            # Verificar headers existentes
            existing_headers = sheet.row_values(1) if sheet.row_values(1) else None
        except Exception as e:
            st.error(f"Error al acceder a la hoja: {str(e)}")
            return

        # Preparar datos para enviar
        data_to_send = {
            'Fecha': fecha_actual,
            'Hora': hora_actual,
            'Nombre_Paciente': str(patient_name),
            # ... [tus otras columnas] ...
            'Prob_Benigno': prob_benigno,
            'Prob_Maligno': prob_maligno
        }

        # Si no hay headers o hay duplicados, recrearlos
        if not existing_headers or len(existing_headers) != len(set(existing_headers)):
            headers = list(data_to_send.keys())
            # Limpiar hoja existente si es necesario
            if existing_headers:
                sheet.clear()
            sheet.insert_row(headers, 1)
        
        # Añadir nueva fila
        new_row = list(data_to_send.values())
        sheet.append_row(new_row)
        
        # Formatear porcentajes
        last_row = len(sheet.get_all_records(expected_headers=list(data_to_send.keys())))
        sheet.format(f"S{last_row}:T{last_row}", {  # Ajusta las letras según tus columnas
            "numberFormat": {"type": "NUMBER", "pattern": "0.00\"%\""}
        })
        
        st.success("✅ Datos guardados correctamente!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
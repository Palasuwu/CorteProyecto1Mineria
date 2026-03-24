import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
# Rutas absolutas para evitar errores de ruta relativa
RUTAS = {
    "MATRIMONIOS": "c:\\Users\\angge\\mineria\\CorteProyecto1Mineria\\data_matrimonios",
    "DIVORCIOS":   "c:\\Users\\angge\\mineria\\CorteProyecto1Mineria\\data_divorcios"
}

# =============================================================================
# 2. FUNCIÓN DE CARGA Y UNIFICACIÓN
# =============================================================================
def cargar_dataset_ine(ruta_carpeta, nombre_dataset):
    print(f"\n🚀 PROCESANDO: {nombre_dataset} desde {ruta_carpeta}...")
    
    archivos_sav = [f for f in os.listdir(ruta_carpeta) if f.endswith('.sav')]
    
    if not archivos_sav:
        print(f"⚠️  ALERTA: No se encontraron archivos .sav en {ruta_carpeta}")
        return None

    lista_dfs = []
    
    # DICCIONARIO DE RENOMBRE (IMPORTANTE):
    # El INE cambia nombres de columnas entre años.
    # Aquí es donde deberás agregar las correcciones si el código falla al unir.
    # Ejemplo: 'DEPTO': 'DEPARTAMENTO'
    mapa_columnas = {
        'A_OCUR': 'ANIO_OCURRENCIA',
        'MES_OCUR': 'MES_OCURRENCIA',
        'DEPTO': 'DEPARTAMENTO',
        'MUN': 'MUNICIPIO'
    }

    for archivo in archivos_sav:
        full_path = os.path.join(ruta_carpeta, archivo)
        try:
            # Usamos convert_categoricals=False para cargar los códigos (números)
            # Esto evita errores si un año dice "Guatemala" y otro "GUATEMALA"
            df_temp = pd.read_spss(full_path, convert_categoricals=False)
            
            # Normalizar nombres de columnas
            df_temp.rename(columns=mapa_columnas, inplace=True)
            
            # Agregar columna de referencia del archivo
            df_temp['ARCHIVO_ORIGEN'] = archivo
            
            lista_dfs.append(df_temp)
            print(f"   -> Cargado: {archivo} ({len(df_temp)} filas)")
        except Exception as e:
            print(f"   ❌ Error leyendo {archivo}: {e}")

    # Unir todos los años
    if lista_dfs:
        df_unificado = pd.concat(lista_dfs, ignore_index=True)
        # Limpieza básica de códigos de error del INE (99, 999 suelen ser 'Ignorado')
        df_unificado.replace([99, 999, 9999], np.nan, inplace=True)
        return df_unificado
    return None

# =============================================================================
# 3. FUNCIÓN DE REPORTE AUTOMÁTICO (EDA)
# =============================================================================
def generar_reporte_avance(df, titulo):
    if df is None: return

    print("\n" + "#"*60)
    print(f" REPORTE DE AVANCE: {titulo}")
    print("#"*60)
    
    # --- A. DESCRIPCIÓN GENERAL  ---
    n_obs, n_vars = df.shape
    print(f"\n1. DIMENSIONES:")
    print(f"   - Total Observaciones (Filas): {n_obs}")
    print(f"   - Total Variables (Columnas): {n_vars}")
    
    print(f"\n2. TIPOS DE VARIABLES (Muestra):")
    print(df.dtypes.head(10)) # Muestra solo las primeras 10 para no saturar
    
    # --- B. VARIABLES NUMÉRICAS  ---
    print(f"\n3. EXPLORACIÓN NUMÉRICA (Tendencia Central y Dispersión):")
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filtramos columnas que no son estadísticas reales (como códigos de mes o archivo)
    cols_num_reales = [c for c in cols_num if 'ARCHIVO' not in c and 'OCUR' not in c]
    
    if cols_num_reales:
        resumen = df[cols_num_reales].describe().T
        resumen['moda'] = df[cols_num_reales].mode().iloc[0]
        print(resumen[['mean', 'std', 'min', '50%', 'max', 'moda']])
        
        # Gráficos rápidos para el PDF
        for col in cols_num_reales[:3]: # Solo graficamos las primeras 3 para probar
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribución de {col} ({titulo})")
            plt.show()
    else:
        print("   No se detectaron variables numéricas obvias para analizar.")

    # --- C. VARIABLES CATEGÓRICAS  ---
    print(f"\n4. EXPLORACIÓN CATEGÓRICA (Top 5 Frecuencias):")
    # Asumimos que las columnas con pocas opciones únicas son categóricas
    cols_cat = [c for c in df.columns if df[c].nunique() < 50 and c not in cols_num_reales]
    
    for col in cols_cat[:5]: # Analizamos solo las primeras 5 detectadas
        print(f"\n   -> Variable: {col}")
        conteo = df[col].value_counts(normalize=True) * 100
        print(conteo.head(5).to_string())

# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

# Cargar y Analizar Matrimonios
df_matrimonios = cargar_dataset_ine(RUTAS["MATRIMONIOS"], "MATRIMONIOS")
generar_reporte_avance(df_matrimonios, "MATRIMONIOS")

# Cargar y Analizar Divorcios
df_divorcios = cargar_dataset_ine(RUTAS["DIVORCIOS"], "DIVORCIOS")
generar_reporte_avance(df_divorcios, "DIVORCIOS")

# Sugerencia de cruce (Opcional)
print("\n" + "="*60)
print("TIP: Para el punto de 'Relaciones entre variables' [cite: 39]")
print("Intenta cruzar la EDAD con el DEPARTAMENTO o la OCUPACIÓN.")
print("="*60)

# =============================================================================
# 5. CREACIÓN DE df_master Y MODELADO PREDICTIVO
# =============================================================================
# Unimos matrimonios y divorcios con etiqueta para predicción
if df_matrimonios is not None and df_divorcios is not None:
    df_matrimonios['DIVORCIO'] = 0
    df_divorcios['DIVORCIO'] = 1
    # Agregamos columna estándar de edad si no existe
    for col in ['EDADHOM', 'EDAD_HOMBRE']:
        if col in df_matrimonios.columns and 'EDAD_HOMBRE' not in df_matrimonios.columns:
            df_matrimonios['EDAD_HOMBRE'] = df_matrimonios[col]
            df_divorcios['EDAD_HOMBRE'] = df_divorcios.get('EDADHOM', df_divorcios.get('EDAD_HOMBRE'))
            break
    for col in ['EDADMUJ', 'EDAD_MUJER']:
        if col in df_matrimonios.columns and 'EDAD_MUJER' not in df_matrimonios.columns:
            df_matrimonios['EDAD_MUJER'] = df_matrimonios[col]
            df_divorcios['EDAD_MUJER'] = df_divorcios.get('EDADMUJ', df_divorcios.get('EDAD_MUJER'))
            break

    df_master = pd.concat([df_matrimonios, df_divorcios], ignore_index=True)

    # Algunas variables derivadas que ya estaban en exploración
    if 'EDAD_HOMBRE' in df_master.columns and 'EDAD_MUJER' in df_master.columns:
        df_master['DIFERENCIA_EDAD'] = (df_master['EDAD_HOMBRE'] - df_master['EDAD_MUJER']).abs()

    # Features mínimos para el modelo
    required_features = ['EDAD_HOMBRE', 'EDAD_MUJER', 'DIFERENCIA_EDAD']
    if all(c in df_master.columns for c in required_features):
        df_model = df_master.dropna(subset=required_features + ['DIVORCIO'])

        X = df_model[required_features]
        y = df_model['DIVORCIO']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        scores = cross_val_score(model, X, y, cv=5)
        print("Cross-validation score promedio:", scores.mean())

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    else:
        print('⚠️ No se encontraron todas las columnas requeridas para modelar:', required_features)
else:
    print('⚠️ No se pudo crear df_master para modelado porque faltan datos de uno de los conjuntos.')
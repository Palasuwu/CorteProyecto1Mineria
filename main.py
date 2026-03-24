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
# Rutas exactas proporcionadas
RUTAS = {
    "MATRIMONIOS": "./data_matrimonios",
    "DIVORCIOS":   "./data_divorcios " # Ajuste: eliminé el espacio extra al final si lo hubiera
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

# Asumiendo df_master ya está limpio (de compañeros)
# Crear variable objetivo: 1 para divorcios, 0 para matrimonios
df_master['DIVORCIO'] = df_master['ARCHIVO_ORIGEN'].str.contains('divorcios').astype(int)

# Seleccionar features
features = ['EDAD_HOMBRE', 'EDAD_MUJER', 'DIFERENCIA_EDAD']
X = df_master[features]
y = df_master['DIVORCIO']

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir y evaluar
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Validación cruzada
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores.mean())

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Matriz de Confusión')
plt.show()
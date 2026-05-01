# Proyecto de Minería de Datos: Predicción de Segundo Matrimonio

## Universidad del Valle de Guatemala  
**Curso:** CC3074 - Minería de Datos  
**Proyecto:** Resultados  
**Tema:** Análisis y modelado predictivo sobre matrimonios y divorcios en Guatemala  

---

## Descripción general del proyecto

Este proyecto tiene como objetivo aplicar técnicas de minería de datos y aprendizaje automático sobre bases de datos relacionadas con matrimonios y divorcios en Guatemala. A partir de los datos proporcionados en formato `.sav`, se realiza un proceso completo de análisis, limpieza, transformación, selección de variable respuesta, entrenamiento de modelos predictivos y evaluación de resultados.

El enfoque principal de esta entrega es la construcción de modelos de clasificación para predecir si un registro corresponde a un caso de **segundo matrimonio**. Esta variable fue seleccionada debido a que permite estudiar patrones sociales y demográficos asociados a personas que contraen matrimonio nuevamente.

---

## Objetivo del proyecto

Desarrollar modelos predictivos capaces de clasificar si una persona o registro corresponde a un segundo matrimonio, utilizando variables disponibles en las bases de datos de matrimonios. Para ello, se aplican técnicas de preprocesamiento, separación de datos en entrenamiento y prueba, entrenamiento de algoritmos de machine learning y evaluación mediante métricas de clasificación.

---

## Estructura del repositorio

La estructura principal del proyecto es la siguiente:

```text
CorteProyecto1Mineria/
│
├── data_divorcios/
│   ├── divorcios_parte1.sav
│   └── divorcios_parte2.sav
│
├── data_matrimonios/
│   ├── matrimonios_2019.sav
│   ├── matrimonios_2020.sav
│   ├── matrimonios_2021.sav
│   ├── matrimonios_2022.sav
│   ├── matrimonios_2023.sav
│   └── matrimonios_extra.sav
│
├── Avances2.ipynb
├── Avances3.ipynb
├── Proyecto.ipynb
├── logistic_regression.ipynb
├── main.py
└── README.md
````

---

## Descripción de las carpetas y archivos

### `data_divorcios/`

Contiene las bases de datos relacionadas con divorcios. Estas bases pueden utilizarse para análisis complementarios o comparativos dentro del proyecto.

Archivos incluidos:

```text
divorcios_parte1.sav
divorcios_parte2.sav
```

### `data_matrimonios/`

Contiene las bases de datos principales utilizadas para el modelado predictivo. Los archivos corresponden a registros de matrimonios de distintos años.

Archivos incluidos:

```text
matrimonios_2019.sav
matrimonios_2020.sav
matrimonios_2021.sav
matrimonios_2022.sav
matrimonios_2023.sav
matrimonios_extra.sav
```

### `logistic_regression.ipynb`

Notebook principal para el modelo de regresión logística. En este archivo se realiza el proceso completo de:

* Carga de archivos `.sav`
* Unión de bases de matrimonios
* Limpieza inicial de datos
* Creación de la variable respuesta
* Separación de variables predictoras y variable objetivo
* División entre entrenamiento y prueba
* Preprocesamiento de variables numéricas y categóricas
* Entrenamiento de varios modelos de regresión logística
* Evaluación con métricas de clasificación
* Matriz de confusión
* Selección del mejor modelo

### `Proyecto.ipynb`, `Avances2.ipynb` y `Avances3.ipynb`

Notebooks utilizados durante las fases previas del proyecto, incluyendo exploración de datos, análisis descriptivo, limpieza y avances del trabajo grupal.

### `main.py`

Archivo auxiliar del proyecto. Puede utilizarse para pruebas, ejecución de funciones o integración de procesos complementarios.

---

## Variable respuesta seleccionada

La variable respuesta seleccionada para el modelo fue:

```text
SEGUNDO_MATRIMONIO
```

Esta variable es de tipo **cualitativa binaria**, ya que representa dos posibles clases:

```text
0 = No corresponde a segundo matrimonio
1 = Sí corresponde a segundo matrimonio
```

La elección de esta variable permite transformar el problema en una tarea de **clasificación supervisada**, donde el objetivo es predecir si un registro pertenece o no a la categoría de segundo matrimonio.

Esta variable fue construida a partir de las columnas disponibles en las bases de matrimonios relacionadas con el estado civil previo o el número de matrimonio de las personas registradas. Dependiendo de la disponibilidad de columnas en los archivos `.sav`, el notebook identifica las variables adecuadas para generar esta clasificación.

---

## Justificación de la variable respuesta

Se eligió la variable `SEGUNDO_MATRIMONIO` porque permite analizar un fenómeno social relevante: la reincidencia matrimonial. Este tipo de variable puede estar relacionada con factores como edad, año del evento, características demográficas, lugar de registro y otros atributos presentes en la base de datos.

Además, al tratarse de una variable categórica binaria, es adecuada para aplicar algoritmos de clasificación como:

* Regresión logística
* Árboles de decisión
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine
* Naive Bayes

En esta parte del proyecto se implementó específicamente el modelo de **regresión logística**.

---

## Modelo implementado: Regresión Logística

La regresión logística es un algoritmo de clasificación supervisada utilizado para predecir variables categóricas binarias. En este proyecto se utilizó para estimar la probabilidad de que un registro corresponda a un segundo matrimonio.

Aunque su nombre contiene la palabra “regresión”, este modelo se utiliza en problemas de clasificación cuando la variable respuesta tiene dos clases. El modelo calcula una probabilidad y luego clasifica cada caso en una de las dos categorías.

---

## Preprocesamiento realizado

Antes de entrenar los modelos, se aplicaron varias transformaciones necesarias para asegurar que los datos fueran compatibles con los algoritmos de machine learning.

### 1. Carga de datos

Los datos se cargaron directamente desde los archivos `.sav` ubicados en la carpeta `data_matrimonios/`. Esto permite que el notebook sea reproducible y no dependa de archivos intermedios externos.

### 2. Unión de bases

Las bases de matrimonios de varios años fueron combinadas en un solo conjunto de datos para aumentar la cantidad de registros disponibles y permitir un análisis más completo.

### 3. Limpieza de columnas

Se eliminaron columnas irrelevantes o problemáticas para el entrenamiento del modelo. También se revisaron valores nulos y tipos de datos.

### 4. Construcción de la variable respuesta

Se creó la variable `SEGUNDO_MATRIMONIO`, codificada como una variable binaria.

### 5. Separación de variables

El conjunto de datos se dividió en:

```text
X = variables predictoras
y = variable respuesta SEGUNDO_MATRIMONIO
```

### 6. División entrenamiento/prueba

Se utilizó una división de datos en entrenamiento y prueba para evaluar el desempeño del modelo con datos no vistos durante el entrenamiento.

Generalmente, se utilizó una proporción como:

```text
80% entrenamiento
20% prueba
```

### 7. Transformación de variables numéricas

Para las variables numéricas se aplicó:

* Imputación de valores faltantes con la mediana
* Escalamiento con `StandardScaler`

### 8. Transformación de variables categóricas

Para las variables categóricas se aplicó:

* Conversión a texto para evitar errores por mezcla de tipos
* Imputación de valores faltantes
* Codificación con `OneHotEncoder`

Durante el desarrollo se corrigió un error relacionado con columnas categóricas que tenían valores mezclados entre texto y números. El error indicaba que el codificador requería entradas uniformes. Para solucionarlo, se convirtió cada columna categórica a tipo `str` antes de aplicar `OneHotEncoder`.

---

## Corrección aplicada al preprocesamiento

Durante la ejecución del notebook `logistic_regression.ipynb`, se presentó el siguiente error:

```text
TypeError: Encoders require their input argument must be uniformly strings or numbers. Got ['float', 'str']
```

Este error ocurrió porque algunas columnas categóricas contenían valores mezclados de tipo numérico y texto. Para solucionarlo, se agregó una conversión explícita de las columnas categóricas a texto antes del entrenamiento:

```python
X_train = X_train.copy()
X_test = X_test.copy()

for col in columnas_categoricas:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
```

Con esta corrección, el `OneHotEncoder` puede procesar correctamente las variables categóricas sin conflictos de tipos.

---

## Configuraciones de modelos probadas

Se entrenaron varias configuraciones de regresión logística con distintos hiperparámetros:

```text
LogReg_1_baseline
LogReg_2_balanceado
LogReg_3_regularizado
LogReg_4_menos_regularizado
```

Las configuraciones variaron principalmente en:

* Valor de `C`
* Uso de `class_weight`
* Nivel de regularización
* Balanceo de clases

Ejemplo de configuraciones utilizadas:

```python
configuraciones = [
    {"nombre": "LogReg_1_baseline", "C": 1.0, "penalty": "l2", "class_weight": None, "solver": "lbfgs"},
    {"nombre": "LogReg_2_balanceado", "C": 1.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs"},
    {"nombre": "LogReg_3_regularizado", "C": 0.1, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs"},
    {"nombre": "LogReg_4_menos_regularizado", "C": 10.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs"},
]
```

---

## Métricas de evaluación

Para evaluar los modelos se utilizaron métricas propias de problemas de clasificación:

### Accuracy

Mide el porcentaje total de predicciones correctas.

### Precision

Indica qué proporción de los casos clasificados como segundo matrimonio realmente pertenecen a esa clase.

### Recall

Indica qué proporción de los casos reales de segundo matrimonio fueron detectados correctamente por el modelo.

### F1-score

Combina precision y recall en una sola métrica. Es útil cuando existe desbalance entre clases.

### ROC-AUC

Evalúa la capacidad del modelo para separar correctamente las clases usando probabilidades.

### Matriz de confusión

Permite observar cuántos casos fueron clasificados correctamente e incorrectamente en cada clase.

---

## Interpretación general de resultados

Los resultados permiten comparar las distintas configuraciones de regresión logística y seleccionar el modelo con mejor desempeño. La elección del mejor modelo no se basa únicamente en el accuracy, ya que en problemas de clasificación con posible desbalance de clases es importante revisar también precision, recall, F1-score y ROC-AUC.

En este proyecto se prioriza especialmente el **F1-score**, ya que balancea la capacidad del modelo para detectar correctamente los segundos matrimonios sin generar demasiados falsos positivos.

Si las clases se encuentran desbalanceadas, los modelos con `class_weight="balanced"` pueden ofrecer mejores resultados porque asignan mayor importancia a la clase minoritaria durante el entrenamiento.

---

## Librerías utilizadas

El proyecto utiliza principalmente las siguientes librerías de Python:

```python
pandas
numpy
pyreadstat
scikit-learn
matplotlib
seaborn
```

Estas librerías permiten cargar archivos `.sav`, manipular datos, construir modelos predictivos, evaluar resultados y generar visualizaciones.

---

## Instalación de dependencias

Para ejecutar el proyecto, se recomienda instalar las dependencias necesarias con:

```bash
pip install pandas numpy pyreadstat scikit-learn matplotlib seaborn
```

En caso de trabajar desde Anaconda o Miniconda, también puede utilizarse:

```bash
conda install pandas numpy scikit-learn matplotlib seaborn
pip install pyreadstat
```

---

## Cómo ejecutar el proyecto

Para ejecutar el notebook principal:

1. Clonar el repositorio:

```bash
git clone https://github.com/Palasuwu/CorteProyecto1Mineria.git
```

2. Entrar a la carpeta del proyecto:

```bash
cd CorteProyecto1Mineria
```

3. Instalar dependencias:

```bash
pip install pandas numpy pyreadstat scikit-learn matplotlib seaborn
```

4. Abrir Visual Studio Code o Jupyter Notebook.

5. Ejecutar el archivo:

```text
logistic_regression.ipynb
```

6. Ejecutar las celdas en orden, desde la carga de datos hasta la evaluación final del modelo.

---

## Reproducibilidad

El notebook fue ajustado para trabajar directamente con los archivos `.sav` originales del repositorio. Esto evita depender de archivos intermedios como:

```text
X_train_final.csv
y_train_final.csv
X_test_final.csv
y_test_final.csv
```

De esta forma, cualquier integrante del grupo o evaluador puede ejecutar el flujo completo desde cero utilizando únicamente las carpetas de datos incluidas en el proyecto.

---

## Aporte realizado en esta entrega

En esta parte del proyecto se trabajó principalmente en la implementación y corrección del modelo de regresión logística. El aporte incluye:

* Revisión de la estructura del proyecto.
* Identificación de que el notebook original dependía de archivos intermedios externos.
* Ajuste del flujo para utilizar directamente los archivos `.sav`.
* Construcción de la variable respuesta `SEGUNDO_MATRIMONIO`.
* Implementación del pipeline de preprocesamiento.
* Entrenamiento de varias configuraciones de regresión logística.
* Corrección del error de tipos mixtos en variables categóricas.
* Evaluación de modelos mediante métricas de clasificación.
* Generación de resultados para análisis y comparación.

---

## Estado actual del proyecto

Actualmente el proyecto cuenta con una base funcional para el entrenamiento de modelos de clasificación. El notebook `logistic_regression.ipynb` permite ejecutar el flujo completo de regresión logística y obtener resultados comparables entre distintas configuraciones.

Como siguientes pasos, se recomienda implementar y comparar otros algoritmos de aprendizaje automático, por ejemplo:

* Árbol de decisión
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine
* Naive Bayes

Esto permitirá cumplir con el requisito de valorar al menos tres algoritmos diferentes y seleccionar el mejor modelo con base en evidencia.

---

## Posibles mejoras futuras

Algunas mejoras que podrían agregarse son:

* Agregar validación cruzada.
* Implementar búsqueda de hiperparámetros con `GridSearchCV`.
* Comparar regresión logística con modelos más complejos.
* Analizar la importancia de variables.
* Generar gráficas comparativas entre modelos.
* Exportar automáticamente tablas de resultados.
* Guardar el mejor modelo entrenado con `joblib`.
* Documentar con mayor detalle los antecedentes investigados.
* Integrar los resultados finales en un informe PDF.

---

## Comandos útiles de Git

Para revisar cambios locales:

```bash
git status
```

Para agregar el notebook y el README:

```bash
git add logistic_regression.ipynb README.md
```

Para crear un commit:

```bash
git commit -m "Agregar modelo de regresion logistica y documentacion"
```

Para subir los cambios:

```bash
git push origin main
```

Si la rama principal se llama `master`, usar:

```bash
git push origin master
```

---

## Conclusión

Este proyecto aplica un flujo completo de minería de datos sobre registros de matrimonios y divorcios, con énfasis en la predicción de segundos matrimonios. La implementación de regresión logística representa una primera aproximación formal al problema de clasificación, permitiendo evaluar el comportamiento de diferentes configuraciones del modelo.

El proceso realizado demuestra la importancia de preparar adecuadamente los datos antes del entrenamiento, especialmente cuando existen variables categóricas, valores faltantes o tipos de datos mixtos. Además, la comparación de métricas permite seleccionar el modelo más adecuado de acuerdo con el objetivo del análisis.

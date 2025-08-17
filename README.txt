=========================================
BIY7121 - Evaluación 3
Predicción de Confort Climático en Australia
=========================================

INTEGRANTE: Víctor Hernández
GRUPO: Clustering

-----------------------------------------
REQUISITOS PARA PROBAR EL SISTEMA
-----------------------------------------

1. Tener instalado Python 3.8+ y pip.
2. Instalar las dependencias necesarias:
   pip install -r requirements.txt
   pip install wordcloud
   pip install folium

3. (Opcional) Si usas entorno virtual, actívalo antes de instalar.

-----------------------------------------
ARCHIVOS INCLUIDOS
-----------------------------------------

- app.py                        # API Flask con el modelo publicado
- modelo_entrenado.pkl          # Modelo serializado (árbol de decisión)
- entrenamiento_modelo.ipynb    # Notebook original de entrenamiento
- index.html                    # Página web para consumir la API
- ejemplo.csv                   # Archivo de prueba para carga en la web
- README.txt                    # Este archivo de instrucciones

-----------------------------------------
CÓMO PROBAR EL SISTEMA
-----------------------------------------

1. Inicia la API ejecutando en terminal:
   python app.py

2. Abre el archivo index.html en tu navegador (puedes abrirlo directamente o servirlo con Flask).

3. En la web, selecciona el archivo df_numeric.csv y súbelo.

4. Espera a que se procesen los datos. Se mostrarán:
   - Primeras filas del dataset procesado
   - Reporte de clasificación
   - Mejores hiperparámetros y score
   - Árbol de decisión (imagen)
   - Distribución de confort climático (imagen)
   - Nube de palabras de importancia de variables (imagen)
   - Mapa interactivo de zonas no confortables

5. Puedes descargar los archivos generados (csv/json) desde la carpeta del proyecto.

-----------------------------------------
NOTAS
-----------------------------------------

- Si tienes problemas con dependencias, revisa que tengas instalados: flask, pandas, scikit-learn, matplotlib, seaborn, wordcloud, folium, joblib, numpy.
- El modelo y las visualizaciones se regeneran cada vez que subes un nuevo archivo CSV.
- El sistema está diseñado para funcionar en localhost, pero puedes desplegarlo en cualquier servidor compatible con Flask.

-----------------------------------------
CONTACTO
-----------------------------------------

Para dudas o problemas, contactar a: seba.hernandezv@duocuc.cl

=========================================

# Predicción de Confort Climático en Australia

Este proyecto implementa una aplicación web interactiva para analizar y predecir el confort climático en Australia usando modelos de minería de datos y visualizaciones avanzadas.

## Características principales
- **Carga de archivos CSV** con datos climáticos.
- **Procesamiento y filtrado** de datos, creación de la variable "EsConfortable".
- **Entrenamiento automático** de un árbol de decisión con búsqueda de hiperparámetros (GridSearchCV).
- **Visualizaciones interactivas**:
  - Gráfica de barras de confort/no confort
  - Árbol de decisión
  - Nube de palabras de importancia de variables
  - Mapa interactivo de zonas no confortables (Folium)
  - Matriz de correlación interactiva (selección de variables desde el frontend)
- **Interfaz web moderna** con fuente Montserrat y fondo personalizado con imagen climática.

## Estructura del proyecto
```
PRUEBA_3/
├── app.py                  # Backend Flask
├── templates/
│   └── index.html          # Frontend principal
├── static/                 # Imágenes y archivos generados
├── image/                  # Imágenes para fondos y recursos estáticos personalizados
│   └── images.jfif         # Imagen de fondo climático
├── venv/                   # Entorno virtual (no subir a GitHub)
├── requirements.txt        # Dependencias
├── README.txt              # Este archivo
├── ...
```

## Instalación y ejecución
1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/TU_USUARIO/TU_REPO.git
   cd TU_REPO
   ```
2. **Crea y activa un entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```
3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Coloca tu imagen de fondo** en la carpeta `image/` (por defecto, `images.jfif`).
5. **Ejecuta la aplicación:**
   ```bash
   python app.py
   ```
6. **Abre tu navegador en:**
   [http://localhost:5000](http://localhost:5000)

## Personalización visual
- **Fuente principal:** Montserrat (Google Fonts)
- **Fondo:** Imagen climática personalizada ubicada en `/image/images.jfif` (puedes cambiar la imagen y el nombre en el CSS de `index.html`).
- Todos los gráficos y la interfaz usan colores y estilos por defecto para máxima compatibilidad y legibilidad.

## Endpoints personalizados
- `/image/<filename>`: Sirve archivos estáticos desde la carpeta `image/` para fondos y recursos personalizados.

## Notas
- Los archivos grandes (datasets, modelos, venv) están excluidos en `.gitignore`.
- Se pueden personalizar la lista de variables para la matriz de correlación desde el frontend o hacerla dinámica desde el backend.

## Autor
- Victor Hernández Vivanco

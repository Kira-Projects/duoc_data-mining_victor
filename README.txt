=========================================
BIY7121 - Evaluación 3
Predicción de Confort Climático en Australia
=========================================

INTEGRANTE: [Tu Nombre y Apellido]
GRUPO: [Nombre del grupo o TeamName]

-----------------------------------------
REQUISITOS PARA PROBAR EL SISTEMA
-----------------------------------------

1. Tener instalado Python 3.8+ y pip.
2. Instalar las dependencias necesarias:
   pip install -r requirements.txt

3. (Opcional) Si usas entorno virtual, actívalo antes de instalar.

-----------------------------------------
ARCHIVOS INCLUIDOS
-----------------------------------------

- app.py                  # API Flask con el modelo publicado
- modelo_entrenado.pkl    # Modelo serializado (árbol de decisión)
- entrenamiento_modelo.ipynb # Notebook original de entrenamiento
- index.html              # Página web para consumir la API
- ejemplo.csv             # Archivo de prueba para carga en la web
- README.txt              # Este archivo de instrucciones

-----------------------------------------
CÓMO PROBAR EL SISTEMA
-----------------------------------------

1. Inicia la API ejecutando en terminal:
   python app.py

2. Abre el archivo index.html en tu navegador (puedes abrirlo directamente o servirlo con Flask).

3. En la web, selecciona el archivo ejemplo.csv y súbelo.

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

Para dudas o problemas, contactar a: [Tu correo o contacto]

========================================= 
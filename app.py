import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import joblib
import matplotlib.colors as mcolors
import shutil
import pickle
import numpy as np
import json
import seaborn as sns
from wordcloud import WordCloud

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/image/<path:filename>')
def image_files(filename):
    return send_from_directory('image', filename)

@app.route('/upload', methods=['POST'])
def upload():
    print("Recibida petición /upload")
    if 'file' not in request.files:
        print("No se envió ningún archivo")
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    file = request.files['file']
    if file.filename == '':
        print("Nombre de archivo vacío")
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    try:
        print("Leyendo CSV...")
        df = pd.read_csv(file)
        print("CSV leído, columnas:", df.columns.tolist())
        # Variables necesarias
        variables = ['MinTemp', 'MaxTemp', 'Humidity3pm', 'WindGustSpeed', 'Sunshine', 'Rainfall', 'Latitude', 'Longitude']
        print("Tipos de datos de las variables:")
        print(df[variables].dtypes)
        print("Filtrando filas sin nulos...")
        df_filtrado = df.dropna(subset=variables)
        print("Filtrado realizado, filas:", len(df_filtrado))

        # Reducir a 10,000 filas si hay más
        if len(df_filtrado) > 10000:
            print("Reduciendo a 10,000 filas para acelerar el entrenamiento...")
            df_filtrado = df_filtrado.sample(n=10000, random_state=42)
            print("Filas tras muestreo:", len(df_filtrado))
            print("Primeras 5 filas tras muestreo:")
            print(df_filtrado.head())

        # Crear variable EsConfortable
        print("Creando variable EsConfortable...")
        df_filtrado['EsConfortable'] = (
            (df_filtrado['MinTemp'] >= 10) &
            (df_filtrado['MaxTemp'] <= 30) &
            (df_filtrado['Humidity3pm'] >= 30) & (df_filtrado['Humidity3pm'] <= 70) &
            (df_filtrado['WindGustSpeed'] <= 40) &
            (df_filtrado['Sunshine'] >= 6) &
            (df_filtrado['Rainfall'] <= 2)
        ).astype(int)
        print("Variable EsConfortable creada.")
        print("Guardando df_filtrado.csv...")
        df_filtrado.to_csv("df_filtrado.csv", index=False)
        print("df_filtrado.csv guardado.")
        print("Generando gráfica de distribución de confort...")
        plt.figure(figsize=(6,4))
        sns.countplot(data=df_filtrado, x='EsConfortable')
        plt.title("Distribución de Confort Climático")
        plt.xticks([0, 1], ["No Confort", "Confort"])
        plt.ylabel("Cantidad de Registros")
        plt.xlabel("Confort Climático")
        plt.tight_layout()
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/distribucion_confort.png', dpi=150, facecolor='white')
        plt.close()
        print("Gráfica de distribución generada.")
        print("Entrenando modelo de árbol de clasificación...")
        filas, columnas = df_filtrado.shape
        head = df_filtrado.head().to_dict(orient='records')
        columnas_nombres = list(df_filtrado.columns)
        
        # Entrenar modelo de árbol de clasificación
        X = df_filtrado[variables]
        y = df_filtrado['EsConfortable']
        
        # División entrenamiento/prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Definir la grilla de hiperparámetros
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 3, 5, 8],
            'min_samples_split': [2, 5, 6, 7, 8, 9, 10],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': [None, 'sqrt', 'log2']
        }
        
        # Crear el clasificador
        arbol = DecisionTreeClassifier(random_state=42)
        
        # Buscar la mejor combinación con validación cruzada
        grid_search = GridSearchCV(arbol, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_
        
        # Entrenar el mejor modelo sobre el conjunto de entrenamiento
        best_model.fit(X_train, y_train)
        
        # Predecir sobre el conjunto de prueba
        y_pred = best_model.predict(X_test)
        
        # Evaluar el modelo
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Guardar el modelo con joblib
        joblib.dump(best_model, 'modelo_arbol.pkl')
        
        # Guardar el modelo con pickle
        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        
        # Copiar/renombrar el modelo a modelo_entrenado.pkl para la entrega
        shutil.copy('best_model.pkl', 'modelo_entrenado.pkl')
        
        # Crear directorio static si no existe
        os.makedirs('static', exist_ok=True)
        
        # Generar y guardar la visualización del árbol con colores por defecto
        plt.figure(figsize=(14, 18))
        plot_tree(
            best_model,
            feature_names=variables,
            class_names=["No Confort", "Confort"],
            filled=True,
            rounded=True,
            fontsize=10,
            impurity=False,
            proportion=True,
            precision=2,
        )
        plt.title("Árbol de Decisión - Confort Climático (Modelo Óptimo)")
        plt.tight_layout()
        plt.savefig('static/arbol_decision.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Generar DataFrame aleatorio con los mismos límites de X
        X_random = pd.DataFrame(np.zeros((X.shape[0], X.shape[1])), columns=X.columns)
        for col in X.columns:
            min_val = X[col].min()
            max_val = X[col].max()
            X_random[col] = np.random.uniform(min_val, max_val, size=X.shape[0])
        X_random.to_csv('data_new.csv', index=False)

        # Convertir el DataFrame a JSON y guardarlo
        data_json = X_random.to_json()
        with open('data_new.json', 'w') as f:
            json.dump(data_json, f)
        
        print("Preparando respuesta JSON...")
        return jsonify({
            'filas': filas,
            'columnas': columnas,
            'columnas_nombres': columnas_nombres,
            'head': head,
            'classification_report': classification_rep,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'arbol_image': '/static/arbol_decision.png'
        })
    except Exception as e:
        return jsonify({'error': f'Error al procesar el archivo: {str(e)}'}), 500

@app.route('/predict_random', methods=['GET'])
def predict_random():
    try:
        # Cargar el modelo
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        # Cargar los datos en formato JSON
        with open('data_new.json', 'r') as f:
            data_json = json.load(f)
        # Convertir el JSON a diccionario
        data_dict = json.loads(data_json)
        # Crear DataFrame
        df_new = pd.DataFrame(data_dict)
        # Hacer predicciones
        y_pred_new = best_model.predict(df_new)
        # Agregar columna de predicción
        df_new['Predict_confortable'] = y_pred_new
        # Convertir el DataFrame a JSON
        df_new_json = df_new.to_json()
        # Obtener los dos primeros registros en JSON
        df_new_json_head2 = df_new.head(2).to_json()
        print(df_new_json_head2)
        # Importancia de características
        importances = best_model.feature_importances_
        columns = df_new.columns[:-1]
        feature_importances_df = pd.DataFrame({'feature': columns, 'importance': importances})
        feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
        top_15_features = feature_importances_df.head(15)
        # Crear word cloud
        set2_colors = list(mcolors.TABLEAU_COLORS.values())
        text = ' '.join(top_15_features['feature'].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Top 15 Características Importantes', color='#1a237e')
        plt.tight_layout()
        plt.savefig('static/wordcloud_importancia.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        # Devolver las primeras filas, las predicciones, el JSON completo, el JSON de los dos primeros y la ruta de la imagen
        return jsonify({
            'predicciones': y_pred_new.tolist(),
            'data': df_new.head().to_dict(orient='records'),
            'df_new_json': df_new_json,
            'df_new_json_head2': df_new_json_head2,
            'wordcloud_url': '/static/wordcloud_importancia.png'
        })
    except Exception as e:
        return jsonify({'error': f'Error al predecir con datos aleatorios: {str(e)}'}), 500

@app.route('/mapa_no_confort', methods=['GET'])
def mapa_no_confort():
    try:
        import folium
        from folium.plugins import MarkerCluster
        # Cargar el modelo
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        # Cargar el dataset original filtrado (usamos el mismo filtrado que para el modelo)
        if not os.path.exists('df_filtrado.csv'):
            return jsonify({'error': 'No se encontró df_filtrado.csv, ejecuta primero el flujo de entrenamiento.'}), 400
        df_filtrado = pd.read_csv('df_filtrado.csv')
        variables = ['MinTemp', 'MaxTemp', 'Humidity3pm', 'WindGustSpeed', 'Sunshine', 'Rainfall', 'Latitude', 'Longitude']
        # Predecir confortabilidad
        df_filtrado['PredichoConfort'] = best_model.predict(df_filtrado[variables])
        # Filtrar zonas NO confortables
        no_confort_df = df_filtrado[df_filtrado['PredichoConfort'] == 0]
        # Crear mapa centrado en Australia
        mapa = folium.Map(location=[-25.0, 135.0], zoom_start=4)
        # Crear agrupación de puntos
        cluster = MarkerCluster().add_to(mapa)
        # Agregar cada punto no confortable al mapa
        for _, row in no_confort_df.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"No Confortable\nTempMin: {row['MinTemp']}°C\nHumedad: {row['Humidity3pm']}%",
                icon=folium.Icon(color='red', icon='exclamation-sign')
            ).add_to(cluster)
        # Guardar el mapa como HTML
        mapa.save('static/mapa_no_confort.html')
        return jsonify({'mapa_url': '/static/mapa_no_confort.html', 'total_no_confort': len(no_confort_df)})
    except Exception as e:
        return jsonify({'error': f'Error al generar el mapa: {str(e)}'}), 500

@app.route('/correlacion', methods=['POST'])
def correlacion():
    try:
        data = request.get_json()
        variables = data.get('variables')
        if not variables or not isinstance(variables, list):
            return jsonify({'error': 'Debes enviar una lista de variables.'}), 400
        if not os.path.exists('df_filtrado.csv'):
            return jsonify({'error': 'No se encontró df_filtrado.csv.'}), 400
        df = pd.read_csv('df_filtrado.csv')
        # Verificar que las variables existen
        for var in variables:
            if var not in df.columns:
                return jsonify({'error': f'La variable {var} no existe en el dataset.'}), 400
        # Calcular matriz de correlación
        corr = df[variables].corr()
        plt.figure(figsize=(len(variables)*1.2, len(variables)*1))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'label': 'Correlación'})
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/correlacion.png', dpi=150, facecolor='white')
        plt.close()
        return jsonify({'correlacion_url': '/static/correlacion.png'})
    except Exception as e:
        return jsonify({'error': f'Error al generar la matriz de correlación: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 
# Desarrollo API: se disponibilizan los datos de la empresa usando el framework FastAPI.
# Se crean  6 funciones para los endpoints que se consumirán en la API,con un decorador por cada una (@app.get(‘/’))

# Se importan las librerias
from typing import Union
from typing import List
import pandas as pd
import numpy as np
import json
# from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Creación de una aplicación FastAPI:
app = FastAPI()

import zipfile

# Es posible incorporar los datasets comprimidos
# Para ello se
df_parte = None
proyecto1_crew = None
proyecto1_movies = None
# Estas variables se utilizarán más adelante para almacenar los datos cargados desde los archivos CSV.

@app.on_event('startup')
async def startup():
    global df_parte, proyecto1_crew, proyecto1_movies
    zip_file = 'Datasets.zip'
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('../Datasets/')  # indica la direccion para descomprimir
    df_parte = pd.read_csv('../Datasets/df_parte.csv')
    proyecto1_crew = pd.read_csv('../Datasets/Proyecto1_crew.csv')
    proyecto1_movies = pd.read_csv('../Datasets/proyecto1_movies.csv')

def extract_data_from_zip(): # para extraer los archivos comprimidos
    return df_parte, proyecto1_crew, proyecto1_movies


'''
'''

# Alternativamente es posible trabajar localmente con los archivos sin comprimir    
# Para ello se cargan los archivos obtenidos en la etapa anterior, en este caso se inicialmente se usaron 2 de ellos:
# movies = pd.read_csv('proyecto1_movies.csv')
# movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
# movies = movies.dropna(subset=['release_date'])
# Y el archivo con información de directores y actores
# credits =  pd.read_csv('proyecto1_crew.csv')

################################################################
#######################################################################
# FUNCION 1:
# def cantidad_filmaciones_mes( Mes ):
# Se ingresa un mes en idioma Español.
# Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.


def cantidad_filmaciones_mes(Mes: str) -> int:
        # Cargar el archivo CSV con los datos
    _, _, proyecto1_movies = extract_data_from_zip()
        # Mapear los nombres de los meses en español a los números correspondientes
    proyecto1_movies['release_date'] = pd.to_datetime(proyecto1_movies['release_date']) #solo si se ejecuta formato zip

    meses = {
        'enero': 1,
        'febrero': 2,
        'marzo': 3,
        'abril': 4,
        'mayo': 5,
        'junio': 6,
        'julio': 7,
        'agosto': 8,
        'septiembre': 9,
        'octubre': 10,
        'noviembre': 11,
        'diciembre': 12
    }
    # Conviertir el nombre del mes a minúsculas y buscar su número correspondiente
    mes_numero = meses.get(Mes.lower())
    # Filtrar las películas que se estrenaron en el mes consultado y obter la cantidad con len
    cantidad = len(proyecto1_movies[proyecto1_movies['release_date'].dt.month == mes_numero])
    return cantidad


'''
# Alternativamente podria usarse el siguiente código
def cantidad_filmaciones_mes(Mes: str) -> int:
    # Mapear los nombres de los meses en español a los números correspondientes
    meses = {
        'enero': 1,
        'febrero': 2,
        'marzo': 3,
        'abril': 4,
        'mayo': 5,
        'junio': 6,
        'julio': 7,
        'agosto': 8,
        'septiembre': 9,
        'octubre': 10,
        'noviembre': 11,
        'diciembre': 12
    }
    # Conviertir el nombre del mes a minúsculas y buscar su número correspondiente
    mes_numero = meses.get(Mes.lower())
    # Filtrar las películas que se estrenaron en el mes consultado y obter la cantidad con len
    cantidad = len(movies[movies['release_date'].dt.month == mes_numero])
    return cantidad
'''

@app.get('/cantidad-filmaciones/{mes}')
def obtener_cantidad_filmaciones(mes: str):
    cantidad = cantidad_filmaciones_mes(mes)
    respuesta = f"{cantidad} es la cantidad de películas que fueron estrenadas en el mes de {mes}"
    return json.dumps({"Respuesta": respuesta}, ensure_ascii=False).encode('utf8')

'''
# Otra alternativa:
@app.get('/cantidad-filmaciones/{mes}')
def obtener_cantidad_filmaciones(mes: str):
    cantidad = cantidad_filmaciones_mes(mes)
    respuesta = f"{cantidad} es la cantidad de películas que fueron estrenadas en el mes de {mes}"
    return {'Respuesta': respuesta}
'''
# Ejemplo:
# http://127.0.0.1:8000/cantidad-filmaciones/mayo
# 3153 junio

################################################################
#######################################################################

################################################################
#######################################################################
# FUNCION 2:
# def cantidad_filmaciones_dia( Dia ):
# Se ingresa un día en idioma Español.
# Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.

def cantidad_filmaciones_dia(Dia: str) -> str:
    _, _, proyecto1_movies = extract_data_from_zip()
    proyecto1_movies['release_date'] = pd.to_datetime(proyecto1_movies['release_date']) #solo si se ejecuta formato zip
    # Como en la funcion anterior mapeamos los dias
    dias = {
        'lunes': 'Monday',
        'martes': 'Tuesday',
        'miércoles': 'Wednesday',
        'jueves': 'Thursday',
        'viernes': 'Friday',
        'sábado': 'Saturday',
        'domingo': 'Sunday'
    }
    # Luego se convierte el nombre del día en español al equivalente en inglés
    dia_ingles = dias.get(Dia.lower())
    if dia_ingles:
        # Filtrar las películas que se estrenaron en el día consultado
        peliculas_dia = proyecto1_movies[proyecto1_movies['release_date'].dt.day_name() == dia_ingles]
        cantidad = len(peliculas_dia)
        return f"{cantidad} cantidad de películas fueron estrenadas en los días {Dia.capitalize()}."
    else:
        return "Día inválido. Por favor, ingrese un día válido en español."

@app.get('/cantidad-filmaciones-dia/{dia}')
def obtener_cantidad_filmaciones_dia(dia: str):
    cantidad = cantidad_filmaciones_dia(dia)
    return json.dumps({"Respuesta": cantidad}, ensure_ascii=False).encode('utf8')

'''
# Alternativamente es posible aplicar este codigo
def cantidad_filmaciones_dia(Dia: str) -> str:
    # Como en la funcion anterior mapeamos los dias
    dias = {
        'lunes': 'Monday',
        'martes': 'Tuesday',
        'miércoles': 'Wednesday',
        'jueves': 'Thursday',
        'viernes': 'Friday',
        'sábado': 'Saturday',
        'domingo': 'Sunday'
    }
    # Luego se convierte el nombre del día en español al equivalente en inglés
    dia_ingles = dias.get(Dia.lower())
    if dia_ingles:
        # Filtrar las películas que se estrenaron en el día consultado
        peliculas_dia = movies[movies['release_date'].dt.day_name() == dia_ingles]
        cantidad = len(peliculas_dia)
        return f"{cantidad} cantidad de películas fueron estrenadas en los días {Dia.capitalize()}."
    else:
        return "Día inválido. Por favor, ingrese un día válido en español."

@app.get('/cantidad-filmaciones-dia/{dia}')
def obtener_cantidad_filmaciones_dia(dia: str):
    cantidad = cantidad_filmaciones_dia(dia)
    return json.dumps({"Respuesta": cantidad}, ensure_ascii=False).encode('utf8')
'''

'''
# Otra alternativa:
@app.get('/cantidad-filmaciones-dia/{dia}')
def obtener_cantidad_filmaciones_dia(dia: str):
    cantidad = cantidad_filmaciones_dia(dia)
    return {'Respuesta': cantidad}
'''
# Ejemplo:
# http://127.0.0.1:8000/cantidad-filmaciones-dia/lunes

################################################################
#######################################################################
# FUNCION 3:
# def score_titulo( titulo_de_la_filmación ):
# Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.

def score_titulo(titulo_de_la_filmacion: str) -> str:
    _, _, proyecto1_movies = extract_data_from_zip()
    pelicula = proyecto1_movies[proyecto1_movies['title'] == titulo_de_la_filmacion]
    if len(pelicula) > 0:
        titulo = pelicula['title'].iloc[0]
        año_estreno = pelicula['release_year'].iloc[0]
        score = pelicula['vote_average'].iloc[0]
        respuesta = f'La película "{titulo}" fue estrenada en el año {año_estreno} con un score/popularidad de {score}.'
        return respuesta
    else:
        return 'No se encontró ninguna película con ese título, vuelva a intentarlo.'

@app.get('/score-titulo/{titulo}')
def obtener_score_titulo(titulo: str):
    respuesta = score_titulo(titulo)
    return json.dumps({"Respuesta": respuesta}, ensure_ascii=False).encode('utf8')


'''
# Inicialmente se habia aplicado el código:
def score_titulo(titulo_de_la_filmacion: str) -> str:
    pelicula = movies[movies['title'] == titulo_de_la_filmacion]
    if len(pelicula) > 0:
        titulo = pelicula['title'].iloc[0]
        año_estreno = pelicula['release_year'].iloc[0]
        score = pelicula['vote_average'].iloc[0]
        respuesta = f'La película "{titulo}" fue estrenada en el año {año_estreno} con un score/popularidad de {score}'
        return respuesta
    else:
        return 'No se encontró ninguna película con ese título, vuelva a intentarlo.'


@app.get('/score-titulo/{titulo}')
def obtener_score_titulo(titulo: str):
    respuesta = score_titulo(titulo)
    return json.dumps({"Respuesta": respuesta}, ensure_ascii=False).encode('utf8')
'''

'''
# Alternativa a lo anterior:
@app.get('/score-titulo/{titulo}')
def obtener_score_titulo(titulo: str):
    respuesta = score_titulo(titulo)
    return {'Respuesta': respuesta}
'''
# Ejemplo:
# http://127.0.0.1:8000/score-titulo/Titanic

################################################################
#######################################################################
# FUNCION 4:
# def votos_titulo( titulo_de_la_filmación ):
# Se ingresa el título de una filmación esperando como respuesta:
# el título, la cantidad de votos y el valor promedio de las votaciones.
# La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario,
# debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.


def votos_titulo(titulo_de_la_filmacion: str) -> str:
    _, _, proyecto1_movies = extract_data_from_zip()
    pelicula = proyecto1_movies[proyecto1_movies['title'] == titulo_de_la_filmacion]
    if len(pelicula) > 0:
        titulo = pelicula['title'].iloc[0]
        votos = pelicula['vote_count'].iloc[0]
        promedio_votos = pelicula['vote_average'].iloc[0]
        if votos >= 2000:
            respuesta = f'La película "{titulo}" fue estrenada en el año {pelicula["release_year"].iloc[0]}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio_votos}.'
        else:
            respuesta = f'La película "{titulo}" no cumple con la condición de tener al menos 2000 valoraciones.'
        return respuesta
    else:
        return 'No se encontró ninguna película con ese título'

@app.get('/votos-titulo/{titulo}')
def obtener_votos_titulo(titulo: str):
    respuesta = votos_titulo(titulo)
    return json.dumps({"Respuesta": respuesta}, ensure_ascii=False).encode('utf8')


'''
# Inicialmente se había aplicado:
def votos_titulo(titulo_de_la_filmacion: str) -> str:
    pelicula = movies[movies['title'] == titulo_de_la_filmacion]
    if len(pelicula) > 0:
        titulo = pelicula['title'].iloc[0]
        votos = pelicula['vote_count'].iloc[0]
        promedio_votos = pelicula['vote_average'].iloc[0]
        if votos >= 2000:
            respuesta = f'La película "{titulo}" fue estrenada en el año {pelicula["release_year"].iloc[0]}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio_votos}.'
        else:
            respuesta = f'La película "{titulo}" no cumple con la condición de tener al menos 2000 valoraciones.'
        return respuesta
    else:
        return 'No se encontró ninguna película con ese título'


@app.get('/votos-titulo/{titulo}')
def obtener_votos_titulo(titulo: str):
    respuesta = votos_titulo(titulo)
    return json.dumps({"Respuesta": respuesta}, ensure_ascii=False).encode('utf8')
'''

'''
# Alternativa de lo anterior:
@app.get('/votos-titulo/{titulo}')
def obtener_votos_titulo(titulo: str):
    respuesta = votos_titulo(titulo)
    return {'Respuesta': respuesta}
'''
# Ejemplo:
# http://127.0.0.1:8000/votos-titulo/Titanic

################################################################
#######################################################################
# FUNCION 5:
# def get_actor( nombre_actor ):
# Se ingresa el nombre de un actor que se encuentre dentro de un dataset
# debiendo devolver el éxito del mismo medido a través del retorno.
# Además, la cantidad de películas que en las que ha participado y el promedio de retorno.
# La definición no deberá considerar directores.

def get_actor(nombre_actor: str):
    _, proyecto1_crew, proyecto1_movies = extract_data_from_zip()
    # Filtrar el dataframe 'credits' por el nombre del actor
    actor_credits = proyecto1_crew[proyecto1_crew['Name_actors'].str.contains(nombre_actor, case=False, na=False)]
    # Obtener la cantidad de películas en las que ha participado el actor
    num_movies = len(actor_credits)
    # Filtrar el dataframe 'movies' por los IDs de las películas en las que ha participado el actor
    actor_movies = proyecto1_movies[proyecto1_movies['id'].isin(actor_credits['id'])]
    # Calcular el retorno promedio del actor
    avg_return = actor_movies['return'].mean()
    # Corregir el cálculo del retorno total para manejar valores no válidos
    total_return = actor_movies['return'].replace([np.inf, -np.inf], np.nan).dropna().sum()
    # Verificar si el valor de total_return es NaN y asignar cero en su lugar
    total_return = total_return if not np.isnan(total_return) else 0
    # Construir el mensaje de retorno
    message = f"El actor '{nombre_actor}' ha participado en {num_movies} películas. "

    if np.isinf(avg_return):
        message += f"No se puede calcular el promedio de retorno."
    else:
        message += f"Ha conseguido un retorno total de {total_return} con un promedio de {avg_return} por filmación."
    return message

@app.get('/nombre_actor/{nombre_actor}')
def obtener_actor(nombre_actor: str):
    respuesta = get_actor(nombre_actor)
    return json.dumps({"Respuesta": respuesta}, ensure_ascii=False).encode('utf8')


'''
# Tambien es posible aplicar:
def get_actor(nombre_actor: str):

    # Filtrar el dataframe 'credits' por el nombre del actor
    actor_credits = credits[credits['Name_actors'].str.contains(nombre_actor, case=False, na=False)]
    # Obtener la cantidad de películas en las que ha participado el actor
    num_movies = len(actor_credits)
    # Filtrar el dataframe 'movies' por los IDs de las películas en las que ha participado el actor
    actor_movies = movies[movies['id'].isin(actor_credits['id'])]
    # Calcular el retorno promedio del actor
    avg_return = actor_movies['return'].mean()
    # Corregir el cálculo del retorno total para manejar valores no válidos
    total_return = actor_movies['return'].replace([np.inf, -np.inf], np.nan).dropna().sum()
    # Verificar si el valor de total_return es NaN y asignar cero en su lugar
    total_return = total_return if not np.isnan(total_return) else 0
    # Construir el mensaje de retorno
    message = f"El actor '{nombre_actor}' ha participado en {num_movies} películas. "

    if np.isinf(avg_return):
        message += f"No se puede calcular el promedio de retorno."
    else:
        message += f"Ha conseguido un retorno total de {total_return} con un promedio de {avg_return} por filmación."
    return message

@app.get('/nombre_actor/{nombre_actor}')
def obtener_actor(nombre_actor: str):
    respuesta = get_actor(nombre_actor)
    return json.dumps({"Respuesta": respuesta}, ensure_ascii=False).encode('utf8')
'''

# Alternativa:
#@app.get('/nombre_actor/{nombre_actor}')

# Ejemplos:
# http://127.0.0.1:8000/nombre_actor/Tom%20Hanks
# http://127.0.0.1:8000/nombre_actor/George%20Clooney
# http://127.0.0.1:8000/nombre_actor/Kirsten%20Dunst
# http://127.0.0.1:8000/nombre_actor/Robin%20Williams

################################################################
#######################################################################
# FUNCION 6:
# def get_director( nombre_director ):
# Se ingresa el nombre de un director que se encuentre dentro de un dataset
# debiendo devolver el éxito del mismo medido a través del retorno.
# Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.

def get_director(nombre_director: str):
    _, proyecto1_crew, proyecto1_movies = extract_data_from_zip()
    proyecto1_movies['release_date'] = pd.to_datetime(proyecto1_movies['release_date'])  # Solo si se ejecuta desde el formato zip
    # Convertir el nombre del director a minúsculas
    nombre_director = nombre_director.lower()
    # Crear una copia del DataFrame "proyecto1_crew" y convertir la columna "director" a minúsculas
    df_director = proyecto1_crew.copy()
    df_director['Director_name'] = df_director['Director_name'].str.lower()
    # Filtrar el DataFrame por el nombre del director
    df_filtered = df_director[df_director['Director_name'] == nombre_director]
    # Realizar una unión (merge) entre los DataFrames "df_filtered" y "proyecto1_movies" utilizando el campo "id"
    df_merged = pd.merge(df_filtered, proyecto1_movies, on='id')
    # Calcular el retorno total
    retorno_total = df_merged['return'].sum()
    # Lista de películas con su detalle
    peliculas = []
    for _, row in df_merged.iterrows():
        pelicula = {
            "Titulo": row['title'],
            "Fecha": row['release_year'],
            "Retorno_pelicula": row['return'],
            "Budget_pelicula": row['budget'],
            "Revenue_pelicula": row['revenue']
        }
        peliculas.append(pelicula)
    respuesta = {
        'director': nombre_director,
        'retorno_total_director': retorno_total,
        'peliculas': peliculas
    }
    return respuesta

@app.get("/director/{nombre_director}")
def obtener_director(nombre_director: str):
    respuesta = get_director(nombre_director)
    return respuesta
'''
@app.get("/director/{nombre_director}")
def obtener_director(nombre_director: str):
    respuesta = get_director(nombre_director)
    return json.dumps(respuesta, ensure_ascii=False).encode('utf8')
'''

'''
# Desde los df iniciales:
def get_director(nombre_director: str):
    # Se convierte la variable a minúsculas
    nombre_director = nombre_director.lower()
    # Union de dataset por el id
    df_director = pd.merge(credits, movies, on='id')
    df_director['director'] = df_director['director'].str.lower()
    # Creación de un df con los datos requeridos
    df_filtered = df_director[df_director['director'] == nombre_director]
    # Calculo el retorno total
    retorno_total = df_filtered['return'].sum()
    # Lista de películas con su detalle
    peliculas = []
    for _, row in df_filtered.iterrows():
        pelicula = {
            "Titulo": row['title'],
            "Anio": row['release_year'],
            "Retorno_pelicula": row['return'],
            "Budget_pelicula": row['budget'],
            "Revenue_pelicula": row['revenue']
        }
        peliculas.append(pelicula)
    respuesta = {
        'director': nombre_director,
        'retorno_total_director': retorno_total,
        'peliculas': peliculas
    }
    return respuesta

@app.get("/director/{nombre_director}")
def obtener_director(nombre_director: str):
    respuesta = get_director(nombre_director)
    return json.dumps(respuesta, ensure_ascii=False).encode('utf8')
'''
#Alternativa:
#@app.get("/director/{nombre_director}")

# Ejemplos:
# http://127.0.0.1:8000/director/James%20Lapine
# http://127.0.0.1:8000/director/Michele%20Josue
# http://127.0.0.1:8000/director/John%20Lasseter

#####################################
############################################


# Cálculo de similitud:
def calcular_similitud():
    df_parte, _, _ = extract_data_from_zip()
    tfidf = TfidfVectorizer()
    overview_features = tfidf.fit_transform(df_parte["overview"].astype(str))
    similarity_matrix = cosine_similarity(overview_features)
    return similarity_matrix

def obtener_recomendaciones(similarity_matrix, selected_movie_title):
    selected_movie_index = df_parte.index[df_parte["title"].str.lower() == selected_movie_title.lower()][0]
    similar_movies_indices = similarity_matrix[selected_movie_index].argsort()[::-1][1:6]
    recommended_movies = df_parte.loc[similar_movies_indices, "title"].tolist()
    return recommended_movies

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str) -> List[str]:
    similarity_matrix = calcular_similitud()
    recommended_movies = obtener_recomendaciones(similarity_matrix, titulo)
    return recommended_movies




'''
# Pueden surgir algunos problemas al intentar cargar el DataFrame completo con 40000 datos
df_Total = pd.read_csv("df_Total.csv")
# Por lo que se puede cargar la fracción de datos
df_parte = pd.read_csv("df_parte.csv")

# Cálculo de similitud
def calcular_similitud():
    tfidf = TfidfVectorizer()
    overview_features = tfidf.fit_transform(df_parte["overview"].astype(str))
    similarity_matrix = cosine_similarity(overview_features)
    return similarity_matrix
'''
'''
# Recomendación para usar el df usado para entrenar el modelo
def obtener_recomendaciones(similarity_matrix, selected_movie_title):
    selected_movie_index = df_parte.index[df_parte["title"] == selected_movie_title][0]
    similar_movies_indices = similarity_matrix[selected_movie_index].argsort()[::-1][1:6]
    recommended_movies = df_parte.loc[similar_movies_indices, "title"].tolist()
    return recommended_movies
'''

'''
def obtener_recomendaciones(similarity_matrix, selected_movie_title):
    selected_movie_index = df_Total.index[df_Total["title"].str.lower() == selected_movie_title.lower()][0]
    similar_movies_indices = similarity_matrix[selected_movie_index].argsort()[::-1][1:6]
    recommended_movies = df_Total.loc[similar_movies_indices, "title"].tolist()
    return recommended_movies

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str) -> List[str]:
    similarity_matrix = calcular_similitud()
    recommended_movies = obtener_recomendaciones(similarity_matrix, titulo)
    return recommended_movies
'''

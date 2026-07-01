#Importamos las basicas librerías de Python para el análisis de datos y visualización
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importamos linear_model y datasets de sklearn
from sklearn import linear_model
from sklearn import datasets

#importamos train_test_split
from sklearn.model_selection import train_test_split

#importamos las metricas
from sklearn.metrics import mean_squared_error, r2_score


#DATASET
df = pd.read_csv("boston.csv")

#revisamos las primeras filas del dataset
df.head()

#vemos la forma del dataset
df.shape

#vemos la informacion del dataset
df.info()

#eliminamos la columna que no es necesaria
df = df.drop(columns=["Unnamed: 0"])

#vemos las medidas descriptivas
df.describe()

#EJERCICIO 2
#Dividimos la muestra en entrenamiento y validacion

#separamos los atributos y el vector objetivo
X = df.drop(columns=["medv"])
y = df["medv"]

#generamos conjuntos de entrenamiento y validacion
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=123
)

#EJERCICIO 3
#Generamos dos modelos de regresion lineal

#modelo con intercepto
modelo_intercepto = linear_model.LinearRegression(fit_intercept=True)

#modelo sin intercepto
modelo_sin_intercepto = linear_model.LinearRegression(fit_intercept=False)

#entrenamos el modelo con intercepto
modelo_intercepto.fit(X_train, y_train)

#entrenamos el modelo sin intercepto
modelo_sin_intercepto.fit(X_train, y_train)

#generamos predicciones con el modelo con intercepto
y_pred_intercepto = modelo_intercepto.predict(X_test)

#generamos predicciones con el modelo sin intercepto
y_pred_sin_intercepto = modelo_sin_intercepto.predict(X_test)

#EJERCICIO 4
#creamos una funcion para reportar las metricas

def report_scores(y_real, y_pred):
    mse = mean_squared_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)

    print("Error Cuadratico Promedio:", mse)
    print("R2:", r2)


#reporte modelo con intercepto
print("Modelo con intercepto")
report_scores(y_test, y_pred_intercepto)

#reporte modelo sin intercepto
print("Modelo sin intercepto")
report_scores(y_test, y_pred_sin_intercepto)

#comentario ejercicio 4
#El mejor modelo sera el que tenga menor Error Cuadratico Promedio y mayor R2.
#Si el modelo con intercepto presenta mejor R2 y menor error, entonces se selecciona ese modelo.

#EJERCICIO 5
#creamos una funcion para obtener las correlaciones con la variable objetivo

def fetch_features(dataframe, target="medv"):
    correlaciones = dataframe.corr()[target].drop(target)
    correlaciones = correlaciones.abs().sort_values(ascending=False)

    return correlaciones


#obtenemos las correlaciones
correlaciones = fetch_features(df, "medv")

#mostramos las correlaciones
print(correlaciones)

#seleccionamos las 6 variables con mayor correlacion
top_6 = correlaciones.head(6)

print("Las 6 variables con mayor correlacion son:")
print(top_6)

#comentario ejercicio 5
#Las 6 variables con mayor correlacion son las que presentan una relacion mas fuerte con medv.
#Estas variables seran utilizadas para refactorizar el modelo predictivo.

#EJERCICIO 6
#Refactorizamos el modelo usando solo las 6 variables con mayor correlacion

#guardamos el nombre de las 6 variables
variables_top_6 = top_6.index.tolist()

print(variables_top_6)

#creamos una nueva matriz de atributos con esas variables
X_refactorizado = df[variables_top_6]

#definimos nuevamente el vector objetivo
y = df["medv"]

#generamos nuevos conjuntos de entrenamiento y validacion
X_train_ref, X_test_ref, y_train_ref, y_test_ref = train_test_split(
    X_refactorizado,
    y,
    test_size=0.33,
    random_state=123
)

#entrenamos el modelo seleccionado
modelo_refactorizado = linear_model.LinearRegression(fit_intercept=True)

modelo_refactorizado.fit(X_train_ref, y_train_ref)

#generamos predicciones
y_pred_ref = modelo_refactorizado.predict(X_test_ref)

#reportamos metricas del nuevo modelo
print("Modelo refactorizado con 6 variables")
report_scores(y_test_ref, y_pred_ref)

#comentario ejercicio 6
#Se comparan las metricas del modelo completo con las del modelo refactorizado.
#Si el R2 se mantiene similar y el error no aumenta mucho, el modelo reducido es una buena opcion.
#Esto permite explicar medv usando menos variables.

#EJERCICIO 7
#Prediccion de casos

#creamos los arrays del peor y mejor escenario
worst_neighbor = np.array([37.9, 12.6, 3.5, 27.7, 187, 0.87]).reshape(1, -1)

best_neighbor = np.array([1.73, 22, 8.7, 0.46, 711, 0.38]).reshape(1, -1)

#realizamos las predicciones con el modelo refactorizado
pred_worst = modelo_refactorizado.predict(worst_neighbor)

pred_best = modelo_refactorizado.predict(best_neighbor)

print("Valor esperado peor escenario:", pred_worst)
print("Valor esperado mejor escenario:", pred_best)


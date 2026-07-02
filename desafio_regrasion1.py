# Ejercicio 1: Preparación del ambiente de trabajo
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

#Data set
df = pd.read_csv("Credit.csv")

#revisamos las primeras filas del dataset
df.head()

#vemos la forma del dataset
df.shape

#vemos la informacion del dataset
df.info()

#eliminamos la columna que no es necesaria
df = df.drop(columns=["Unnamed: 0"])

#revisamos la columna de Ethnicity 
print(df["Ethnicity"].value_counts())
print(df["Gender"].value_counts())
print(df["Student"].value_counts())
print(df["Married"].value_counts())

#binarisamos la columna de Ethnicity 
df["African_American"] = np.where(df["Ethnicity"] == "African American", 1, 0)
df["Asian"] = np.where(df["Ethnicity"] == "Asian", 1, 0)

#binarizamos todo lo que queda
df["Male"] = np.where(df["Gender"]=="Male",1,0)
df["Student"] = np.where(df["Student"]=="Yes",1,0)
df["Married"] = np.where(df["Married"]=="Yes",1,0)

def plot_histogram(df, column):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True, bins=30)

    media = df[column].mean()
    mediana = df[column].median()

    plt.axvline(media, color='purple', linestyle='--', label=f'Media: {media:.2f}')
    plt.axvline(mediana, color='blue', linestyle=':', label=f'Mediana: {mediana:.2f}')

    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()
    
plot_histogram(df, "Balance")
plot_histogram(df, "Income")
plot_histogram(df, "Cards")
plot_histogram(df, "Rating")

#Como se puede apreciar en los histogramas, las variables Balance, Income y Rating presentan una distribución sesgada a la derecha, mientras que la variable Cards tiene una distribución más uniforme.
#Esto sugiere que algunas de estas variables pueden requerir transformaciones para mejorar la normalidad antes de aplicar modelos de regresión lineal.
#Ejercicio 3: regresión lineal simple

# Variable independiente
C = df[["Student"]]

# Variable dependiente
Z = df["Balance"]

# División entrenamiento y prueba
C_train, C_test, Z_train, Z_test = train_test_split(
    C, Z, test_size=0.2, random_state=42
)

# Modelo 1 
modelo = linear_model.LinearRegression()
modelo.fit(C_train, Z_train)

# Predicciones
Z_pred = modelo.predict(C_test)

# Resultados
print("Intercepto:", modelo.intercept_)
print("Coeficiente:", modelo.coef_[0])

print("R^2:", r2_score(Z_test, Z_pred))
print("MSE:", mean_squared_error(Z_test, Z_pred))

#modelo 2: regresión lineal múltiple
# Variable independiente
C = df[["Income"]] 

# Variable dependiente
Z = df["Balance"]

# División entrenamiento y prueba
C_train, C_test, Z_train, Z_test = train_test_split(
    C, Z, test_size=0.2, random_state=42
)

# Modelo
modelo = linear_model.LinearRegression()
modelo.fit(C_train, Z_train)

# Predicciones
Z_pred = modelo.predict(C_test)

# Resultados
print("Intercepto:", modelo.intercept_)
print("Coeficiente:", modelo.coef_[0])

print("R^2:", r2_score(Z_test, Z_pred))
print("MSE:", mean_squared_error(Z_test, Z_pred))

#Tercer modelo: regresión lineal múltiple con varias variables independientes
# Variable independiente
C = df[["Rating"]]

# Variable dependiente
Z = df["Balance"]

# División entrenamiento y prueba
C_train, C_test, Z_train, Z_test = train_test_split(
    C, Z, test_size=0.2, random_state=42
)

# Modelo
modelo = linear_model.LinearRegression()
modelo.fit(C_train, Z_train)

# Predicciones
Z_pred = modelo.predict(C_test)

# Resultados
print("Intercepto:", modelo.intercept_)
print("Coeficiente:", modelo.coef_[0])

print("R2:", r2_score(Z_test, Z_pred))
print("MSE:", mean_squared_error(Z_test, Z_pred))

#impeccionando comportaminetos 
sns.lmplot(data=df, x="Rating", y="Balance", height=5, aspect=1.3)

plt.title("Balance vs Rating")
plt.show()

sns.lmplot(data=df, x="Income", y="Balance", height=5, aspect=1.3)

plt.title("Balance vs Income")
plt.show()

sns.lmplot(data=df, x="Limit", y="Balance", height=5, aspect=1.3)

plt.title("Balance vs Limit")
plt.show()

#Los graficos muestran una relación positiva entre las variables Rating, Income y Limit con la variable Balance. 
# Esto sugiere que a medida que aumentan estas variables independientes, también tiende a aumentar el Balance.
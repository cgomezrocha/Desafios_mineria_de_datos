#Importamos las basicas librerías de Python para el análisis de datos y visualización
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importamos missingno para revisar datos perdidos
import missingno as msngo

#importamos librerias para analisis factorial
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

#importamos statsmodels para regresion
import statsmodels.api as sm
import statsmodels.formula.api as smf

#DATASET
df = pd.read_csv("bfi.csv")

#revisamos las primeras filas del dataset
df.head()

#DESAFIO 3
#separamos la bateria de preguntas

preguntas = [
    "A1", "A2", "A3", "A4", "A5",
    "C1", "C2", "C3", "C4", "C5",
    "E1", "E2", "E3", "E4", "E5",
    "N1", "N2", "N3", "N4", "N5",
    "O1", "O2", "O3", "O4", "O5"
]

df_preguntas = df_clean[preguntas]

#revisamos las primeras filas
df_preguntas.head()

#reporte de medias
medias = df_preguntas.mean().sort_values()

plt.figure(figsize=(8,8))
plt.plot(medias.values, medias.index, "o")
plt.axvline(medias.mean(), linestyle="--", label="Media general")
plt.title("Media de cada pregunta")
plt.xlabel("Media")
plt.ylabel("Pregunta")
plt.legend()
plt.show()

#matriz de correlaciones
corr = df_preguntas.corr()

plt.figure(figsize=(14,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Matriz de correlaciones")
plt.show()

#comentario desafio 3
#En la matriz de correlaciones se pueden observar grupos de preguntas relacionadas entre si.
#Las preguntas que pertenecen a una misma dimension de personalidad tienden a correlacionarse.
#Por ejemplo, las preguntas A se relacionan con amabilidad, las C con responsabilidad,
#las E con extroversion, las N con neuroticismo y las O con apertura.

#DESAFIO 4
#prueba de esfericidad de Bartlett

chi_square_value, p_value = calculate_bartlett_sphericity(df_preguntas)

print("Chi cuadrado:", chi_square_value)
print("P-value:", p_value)

#prueba KMO
kmo_all, kmo_model = calculate_kmo(df_preguntas)

print("KMO general:", kmo_model)

#comentario pruebas
#Si el p-value de Bartlett es menor a 0.05, la matriz de correlaciones no es identidad.
#Por lo tanto, tiene sentido aplicar analisis factorial.
#El KMO indica si la muestra es adecuada para analisis factorial.
#Valores cercanos o mayores a 0.7 son considerados aceptables.

#instanciamos el modelo factorial con 10 factores y sin rotacion
factor = FactorAnalyzer(
    n_factors=10,
    rotation=None
)

#entrenamos el modelo
factor.fit(df_preguntas)

#extraemos eigenvalues
ev, v = factor.get_eigenvalues()

#scree plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(ev)+1), ev, marker="o")
plt.axhline(1, linestyle="--")
plt.title("Scree Plot")
plt.xlabel("Cantidad de factores")
plt.ylabel("Eigenvalue")
plt.show()

#comentario scree plot
#Se seleccionan los factores que tienen eigenvalue mayor a 1.
#En esta base normalmente se esperan 5 factores, asociados a las cinco dimensiones de personalidad.

#refactorizamos el modelo con 5 factores y rotacion varimax
factor_final = FactorAnalyzer(
    n_factors=5,
    rotation="varimax"
)

#entrenamos el modelo final
factor_final.fit(df_preguntas)

#extraemos las cargas factoriales
cargas = pd.DataFrame(
    factor_final.loadings_,
    index=df_preguntas.columns,
    columns=["Factor_1", "Factor_2", "Factor_3", "Factor_4", "Factor_5"]
)

#visualizamos las cargas
plt.figure(figsize=(10,8))
sns.heatmap(cargas, annot=True, cmap="coolwarm", center=0)
plt.title("Cargas factoriales")
plt.show()

#identificamos los items mas asociados a cada factor
for columna in cargas.columns:
    print(columna)
    print(cargas[columna].abs().sort_values(ascending=False).head(5))
    print("<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3")
    #comentario desafio 4
#Cada factor representa una dimension latente de personalidad.
#Los items con mayor carga en cada factor ayudan a interpretar su significado.
#Se espera encontrar factores asociados a:
#Amabilidad, Escrupulosidad, Extroversion, Neuroticismo y Apertura.

#DESAFIO 5
#extraemos los puntajes factoriales

factor_scores = factor_final.transform(df_preguntas)

factor_scores = pd.DataFrame(
    factor_scores,
    columns=["Factor_1", "Factor_2", "Factor_3", "Factor_4", "Factor_5"]
)

#unimos los puntajes factoriales con las variables demograficas
df_factor = pd.concat(
    [
        df_clean[["gender", "education", "age"]].reset_index(drop=True),
        factor_scores
    ],
    axis=1
)

df_factor.head()

#graficamos la densidad de cada factor

for columna in ["Factor_1", "Factor_2", "Factor_3", "Factor_4", "Factor_5"]:
    plt.figure(figsize=(7,4))
    sns.kdeplot(df_factor[columna], fill=True)
    plt.title(f"Densidad de {columna}")
    plt.show()
    
#modelamos cada factor en funcion de gender, education y age

modelo_factor_1 = smf.ols("Factor_1 ~ gender + education + age", data=df_factor).fit()
modelo_factor_2 = smf.ols("Factor_2 ~ gender + education + age", data=df_factor).fit()
modelo_factor_3 = smf.ols("Factor_3 ~ gender + education + age", data=df_factor).fit()
modelo_factor_4 = smf.ols("Factor_4 ~ gender + education + age", data=df_factor).fit()
modelo_factor_5 = smf.ols("Factor_5 ~ gender + education + age", data=df_factor).fit()

print(modelo_factor_1.summary())
print(modelo_factor_2.summary())
print(modelo_factor_3.summary())
print(modelo_factor_4.summary())
print(modelo_factor_5.summary())

#comentario desafio 5
#Para interpretar cada modelo se revisan los p-values y los coeficientes.
#Si una variable tiene p-value menor a 0.05, se puede considerar significativa.
#El signo del coeficiente indica si aumenta o disminuye el puntaje del factor.
#Por ejemplo, si age tiene coeficiente positivo, a mayor edad aumenta ese factor.
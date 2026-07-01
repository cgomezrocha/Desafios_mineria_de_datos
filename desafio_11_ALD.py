#Importamos las basicas librerías de Python para el análisis de datos y visualización.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importamos de preprocessing
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder

#IMPORTAMOS DE MODEL_SELECTION
from sklearn.model_selection import train_test_split, cross_val_score

#importamos de METRICS
from sklearn.metrics import classification_report

#importamos de ENSEMBLE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from imblearn.over_sampling import SMOTE

#DATASET
df = pd.read_csv('breast_cancer.csv')
df.head() #revisamos las primeras filas del dataset
df.shape #vemos la forma del dataset
df.info() #vemos la información del dataset
df.describe() #vemos la descripción del dataset
df.isnull().sum() #vemos si hay valores nulos en el dataset
df.columns #vemos las columnas del dataset

#eliminamos las columnas que no son necesarias para el análisis
df = df.drop(columns=["id","Unnamed: 32"])

#analisamos la distribucion de los datos
df['diagnosis'].value_counts()

#veamos su distribucion en un histograma
df.hist(figsize=(18,15))
plt.tight_layout()
plt.show()

#codificamos sus variables categoricas
encoder = LabelEncoder()

df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

#verificamos la codificacion
df["diagnosis"].value_counts()

#separaremos en dos variables Z e C
Z = df.drop(columns=["diagnosis"])
C = df["diagnosis"]

#escalamos los daTos
scaler = StandardScaler()
Z = scaler.fit_transform(Z)

#separamos en entrenamiento y validacion
Z_train, Z_test, C_train, C_test = train_test_split(Z, C, test_size=0.3, random_state=123, stratify=C)

#entrenamos el modelo de LDA
lda = LinearDiscriminantAnalysis()
lda.fit(Z_train, C_train)

#hacemos predicciones
C_pred = lda.predict(Z_test)

#reportamos las metricas de desempeño
print(classification_report(C_test, C_pred))

# Instanciamos la clase
oversampler = SMOTE(random_state=11238, sampling_strategy='minority')
# generamos el eversampling de la matriz de entrenamiento y
Z_train_oversamp, C_train_oversamp = oversampler.fit_resample(Z_train, C_train)

#verificamos si se balanceo 
print("Antes de SMOTE:")
print(C_train.value_counts())

print("\nDespués de SMOTE:")
print(pd.Series(C_train_oversamp).value_counts())

#fialmente entrenamos el modelo de QDA con los datos balanceados
qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda.fit(Z_train_oversamp, C_train_oversamp)

#predeciomos 
C_pred_qda = qda.predict(Z_test)

#y evaluamos el desempeño del modelo
print(classification_report(C_test, C_pred_qda))
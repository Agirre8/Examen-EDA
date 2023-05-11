# %%
#Para conseguir un dataset con una dimensión reducidad, aplica la técnica de Selección de variables basada en árbol de decisión mediante 
#las importancias de cada variable (Decision Trees Importances):

#Filtra el tablón para quedarnos solamente con las variables que aglutinan hasta el 95% de la información que se requiere para estimar la variable objetivo.
#random_state=100
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.tree import DecisionTreeRegressor



# %%
datos = pd.read_csv('covtype.data')

# %%
columnas_Wilderness_Area = ["Wilderness_Area" + str(i) for i in range(1, 5)]
columnas_Soil_Type = ["Soil_Type" + str(i) for i in range(1, 41)]
columnas_restantes = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
"Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
"Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
columna_ultima = ["Cover_Type"]
columnas_final = columnas_restantes + columnas_Wilderness_Area + columnas_Soil_Type + columna_ultima


# %%
datos.columns = columnas_final


# %%
#datos.to_csv('covtype_1.data', index=False)

# %%
#EJERCICIO 1
X = datos.drop('Cover_Type', axis=1)
y = datos['Cover_Type']

# %%
#Crear un modelo de árbol de decisión y ajustarlo a los datos:
tree_model = DecisionTreeRegressor(random_state=100)
tree_model.fit(X, y)

# %%
#Obtener las importancias de las variables:
importances = pd.Series(tree_model.feature_importances_, index=X.columns)

# %%
#Ordenar las importancias de las variables de mayor a menor:
sorted_importances = importances.sort_values(ascending=False)

# %%
#Calcular la suma acumulada de las importancias y el porcentaje que representa cada variable:
cumulative_importances = sorted_importances.cumsum()
cumulative_importances_percent = 100*cumulative_importances/cumulative_importances[-1]

# %%
#Seleccionar las variables que aglutinan hasta el 95% de la información requerida:
selected_variables = cumulative_importances_percent[cumulative_importances_percent <= 95].index
selected_variables = selected_variables.append(pd.Index(["Cover_Type"]))

print(selected_variables)

# %%
datos_1 = datos[selected_variables]
print(datos_1)

# %%
#EJERCICIO2

# Generar histograma de todas las variables
plt.hist(datos_1.values, bins=50)
plt.show()


# %%
# Normalizar todas las columnas excepto la última
cols_to_normalize = datos_1.columns[:-1]
datos_1[cols_to_normalize] = (datos_1[cols_to_normalize] - datos_1[cols_to_normalize].min()) / (datos_1[cols_to_normalize].max() - datos_1[cols_to_normalize].min())
datos_1['Cover_Type'] = datos_1['Cover_Type'] - 1 

# %%
datos_norm = datos_1.copy()

# %%
#EJERCICIO3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(datos_norm.iloc[:, :-1], datos_norm.iloc[:, -1], test_size=0.2, random_state=100)

# entrenar modelo de regresión logística
model = LogisticRegression(max_iter=1000, random_state=100)
model.fit(X_train, y_train)



# %%
# predecir en datos de test y calcular métricas de evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
report = classification_report(y_test, y_pred, zero_division=0)
matrix = confusion_matrix(y_test, y_pred)

# imprimir resultados
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Classification report:\n", report)
print("Confusion matrix:\n", matrix)

# %%
#EJERCICIO 4

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Instanciar el modelo de árbol de decisión
tree = DecisionTreeClassifier(random_state=100)

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datos_1.iloc[:, :-1], datos_1.iloc[:, -1], test_size=0.2, random_state=100)

# Entrenar el modelo en los datos de entrenamiento
tree.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = tree.predict(X_test)


# %%
# Calcular las métricas de evaluación del modelo
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
report = classification_report(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Imprimir las métricas de evaluación del modelo
print("Accuracy:", acc)
print("F1-score (weighted):", f1)
print("Classification report:\n", report)
print("Confusion matrix:\n", cm)


# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Definir rango de profundidades
depths = range(2, 31)

# Crear modelo Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=100)

# Definir parámetros de GridSearchCV
param_grid = {'max_depth': depths}

# Realizar GridSearchCV
grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Guardar resultados de GridSearchCV
results = grid_search.cv_results_

# Obtener los valores de los parámetros y el score
params = results['params']
scores = results['mean_test_score']

# Crear gráfica de curva de complejidad
plt.plot(depths, scores, '-o')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.title('Curva de complejidad del modelo Decision Tree')
plt.xticks(depths)
plt.show()

# Obtener mejor valor de profundidad del árbol
best_depth = grid_search.best_params_['max_depth']
print('El valor óptimo de la profundidad del árbol es:', best_depth)


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv, train_sizes=np.linspace(.1, 1.0, 10)):
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Curva de Aprendizaje")
    plt.xlabel("Número de Muestras")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.show()
    
# Generamos la curva de aprendizaje para el modelo Decision Tree Classifier con profundidad óptima
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=100)
dtc = DecisionTreeClassifier(max_depth=12, random_state=100)
plot_learning_curve(dtc, X, y, cv)


# %%
#EJERCICIO5

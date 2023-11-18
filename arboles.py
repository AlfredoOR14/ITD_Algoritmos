from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report 
# información sobre tres especies de flores (setosa, versicolor y virginica).
# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de Árbol de Decisión
modelo_arbol = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
modelo_arbol.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predicciones = modelo_arbol.predict(X_test)

# Evaluar la precisión del modelo
precision = accuracy_score(y_test, predicciones)
print(f'Precisión del modelo: {precision:.2f}')

# Mostrar el informe de clasificación
print('\nInforme de clasificación:')
print(classification_report(y_test, predicciones))
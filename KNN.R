
# Instalamos y cargamos la biblioteca necesaria
install.packages("class")
library(class)

# Cargamos el conjunto de datos iris
data(iris)

# Dividimos el conjunto de datos en entrenamiento y prueba
set.seed(123)
indices_entrenamiento <- sample(1:nrow(iris), nrow(iris)*0.7)
datos_entrenamiento <- iris[indices_entrenamiento, ]
datos_prueba <- iris[-indices_entrenamiento, ]

# Definimos la función KNN
knn_prediccion <- function(datos_entrenamiento, datos_prueba, k) {
  # Realizamos la predicción utilizando KNN
  prediccion <- knn(train = datos_entrenamiento[, -5], 
                    test = datos_prueba[, -5], 
                    cl = datos_entrenamiento$Species, 
                    k = k)
  return(prediccion)
}

# Especificamos el valor de k (número de vecinos)
k <- 3

# Realizamos la predicción
prediccion <- knn_prediccion(datos_entrenamiento, datos_prueba, k)

# Evaluamos la precisión del modelo
precision <- sum(prediccion == datos_prueba$Species) / nrow(datos_prueba)
cat("Precisión del modelo KNN con k =", k, ":", precision, "\n")













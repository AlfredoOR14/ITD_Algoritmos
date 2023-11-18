# librerias construir y entredar el modelo 
import tensorflow as tf 
import numpy as np

# Datos de entrada
# Aca defininos los arrays usando np(Numpy) 
soles = np.array([100, 200, 300, 400, 500], dtype=float)
dolares = np.array([25.98, 51.97, 77.95, 103.93, 129.91], dtype=float)

# Definición del modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Compilación del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamiento del modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(soles, dolares, epochs=1000, verbose=False)
print("Modelo entrenado!")

# Realizar una predicción con el modelo entrenado
nueva_cantidad_soles = 90.0
resultado = modelo.predict([nueva_cantidad_soles])[0][0]

# Imprimir el resultado de la predicción
print(f"Hagamos una predicción para {nueva_cantidad_soles} soles.")
print(f"El resultado es aproximadamente {resultado:.2f} dólare  s.")
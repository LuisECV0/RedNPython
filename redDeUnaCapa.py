#con una sola capa
import tensorflow as tf
import numpy as np
celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

capa = tf.keras.layers.Dense(units = 1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("comenzando entrenemiento ...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000,verbose=False)
print("modelo entrenado")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel(" Magnitud de perdida")
plt.plot(historial.history["loss"])

print("hagamos una prediccion")
resultado = modelo.predict([100.0])
print("el resultado es " + str(resultado) + "fahrenheit")

print("variables internas del modelo")
print((capa.get_weights()))

#variables internas del modelo
#[array([[1.7981852]], dtype=float32), array([31.924826], dtype=float32)]


#con 2 neuronas
import tensorflow as tf
import numpy as np
celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1,oculta2,salida])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("comenzando entrenemiento ...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000,verbose=False)
print("modelo entrenado")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel(" Magnitud de perdida")
plt.plot(historial.history["loss"])

print("hagamos una prediccion")
resultado = modelo.predict([100.0])
print("el resultado es " + str(resultado) + "fahrenheit")

print("variables internas del modelo")
print((capa.get_weights()))
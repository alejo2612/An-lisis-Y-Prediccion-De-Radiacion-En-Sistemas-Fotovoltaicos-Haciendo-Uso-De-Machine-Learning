"""
@author: Alejandro Herrera Jiménez
Proyecto: Análisis Y Predicción De Radiación En Sistemas Fotovoltaicos Haciendo Uso De Machine Learning
"""

#Librerias necesarias para importar
import tensorflow as tf
from keras_tuner.tuners import BayesianOptimization
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Lectura del dataset en dataframe

dataset = pd.read_csv("dataset.csv", delimiter = ";", encoding='ISO-8859-1')

#Datos usados para predecir (timestamp, humedad relativa, radiación metereologica, temperatura de los modulos, temperatura ambiente
X = dataset.iloc[:, 0:9].values
#Datos que se predicen (Radiación global horizontal)
y = dataset.iloc[:, 9].values

#Separación entre datos de entrenamiento y datos de preuba
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Preprocesamineto de los datos
sX = StandardScaler()
X_train = sX.fit_transform(X_train)
X_test = sX.transform(X_test)

#Definición de la función para la busqueda Bayesiana
def model_builder(hp):
  clf = Sequential()

  #Definición de las iteraciones de la cantidad de neuronas por capa oculta
  hp_LSTM_1 = hp.Int('LSTM_1', min_value=1, max_value=1000, step=100)
  
  hp_LSTM_2 = hp.Int('LSTM_2', min_value=1, max_value=1000, step=100)

  hp_LSTM_3 = hp.Int('LSTM_3', min_value=1, max_value=1000, step=100)
  
  #Iteración del Dropout
  hp_Dropout = hp.Float('Dropout_1', min_value=0.0, max_value=0.5, step=0.1)
  
  #Iteración de learning rate
  hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2)
  
  #Preentrenamiento de prueba para la busqueda Bayesiana
  
  clf.add(LSTM(units=hp_LSTM_1, activation='relu',input_shape = (X_train.shape[1], 1), return_sequences=True))
  
  clf.add(LSTM(units=hp_LSTM_2, activation='relu', return_sequences=True))
  
  clf.add(LSTM(units=hp_LSTM_3, activation='relu'))
  clf.add(Dropout(hp_Dropout))
  clf.add(Dense(units = 1))
  
  # Compilación del modelo con Adam y MSE
  clf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['MSE'])
  
  return clf

#Se establecen los parametros de la busqueda Bayesiana
tuner = BayesianOptimization(model_builder, 
                                objective='val_loss',
                                max_trials=15)

tuner.search(X_train, Y_train, epochs=20, validation_split=0.2)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

#Una vez seleccionados los mejores Hiperparámetros se realiza el entrenamiento final del modelo
clf = tuner.hypermodel.build(best_hps)
#En este caso tiene epoch de 40 pero se puede variar
history = clf.fit(X_train, Y_train, epochs=40)

#Se guarda el modelo como un archivo .h5
clf.save('modelo_entrenado')

#Se predicen los datos con el modelo entrenado
y_pred = clf.predict(X_test)

#Se observan los resultados
# En esta sección se observan e imprimen los resultados de la Busqueda Bayesiana
learning_rate = best_hps.get('learning_rate')
LSTM1 = best_hps.get('LSTM_1')
LSTM2 = best_hps.get('LSTM_2')
LSTM3 = best_hps.get('LSTM_3')
Dopout_1 = best_hps.get('Dopout_1')
print(learning_rate)
print(LSTM1)
print(LSTM2)
print(LSTM3)
print(Dopout_1)


#Graficación de resultados

#Grafica de error relativo (no es muy diciente para este caso)
error = []
for i in range(len(y_pred)):
    val1= (y_pred[i])
    val2= (Y_test[i])
    if Y_test[i] != 0:
        a = ((val1-val2)/val2)*100
        error.append(a)
 
    
plt.plot(error)
plt.xlabel('#datos')
plt.ylabel('% error')
plt.title('Error relativo')
plt.show()

#Grafica de compración entre los datos predecidos y los datos reales
primeros_p = []
primeros_t = []
e = 0
#Se recomienda usar solo 100 datos para poder observar de mejor manera la grafica
while e<100:
    primeros_p.append(y_pred[e])
    primeros_t.append(Y_test[e])
    e +=1
    
plt.plot(primeros_p, label="y_pred")
plt.plot(primeros_t, label="Y_test")    
plt.xlabel('#datos')
plt.ylabel('Radiación [w/m^2]')
plt.title('Comparación datos reales vs predecidos')
plt.legend()
plt.show()


#Grafica de correlación (es la mejor manera de observar que tan bien quedó el modelo)
plt.scatter(Y_test, y_pred)
plt.xlabel('Datos reales')
plt.ylabel('Datos predecidos')
plt.title('Correlación entre datos reales y datos predecidos')
r, _ = stats.pearsonr(Y_test, y_pred)
plt.legend(["R = 0.983"])
plt.show()


# En esta parte se hace la predicción de los datos futuros
#Tener en cuenta que cada paso temporal son 15 minutos
look_back = 10 # Cantidad de pasos anteriores para predecir el siguiente
last_sequence = X_test[-look_back]
num_steps = 15 #Cantidad de pasos temporales siguientes
predicted_values = []
for _ in range(num_steps):    
    next_value = clf.predict(np.array([last_sequence]))
    predicted_values.append(next_value)
    last_sequence = np.concatenate([last_sequence[1:], next_value.flatten()])
    
print(predicted_values)








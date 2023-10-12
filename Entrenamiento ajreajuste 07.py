import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense #,Dropout
import matplotlib.pyplot as plt
import joblib
from keras import optimizers
from keras.callbacks import EarlyStopping
#from keras import regularizers


df = pd.read_csv('direccion csv', sep=";") # almacenamiento en datagrama
#df = pd.read_csv('./dataset1.1.csv', sep=";")
print(df.head(5))    
dataset = df.values
#X = dataset[:,0:11] # Atributos de entrada
X = dataset[:,1:11] # Atributos de entrada
Y = dataset[:,11] # Etiquetas
#Con relacion al sexo
#0 corresponde a Male
#1 corresponde a Femal e 
#X[:,1] = [0 if value=="Male" else value for value in X[:,1]]
#X[:,1] = [1 if value=="Female" else value for value in X[:,1]]
#X[:,1] = [2 if value=="Other" else value for value in X[:,1]]
X[:,0] = [0 if value=="Male" else value for value in X[:,0]]
X[:,0] = [1 if value=="Female" else value for value in X[:,0]]
X[:,0] = [2 if value=="Other" else value for value in X[:,0]]
 
#Con relacion a si es casado
#0 corresponde a NO
#1 corresponde a YES 
#X[:,5] = [0 if value=="No" else value for value in X[:,5]]
#X[:,5] = [1 if value=="Yes" else value for value in X[:,5]]
X[:,4] = [0 if value=="No" else value for value in X[:,4]]
X[:,4] = [1 if value=="Yes" else value for value in X[:,4]]

#Con relacion al tipo de trabajo
#0 corresponde a Never_worked
#1 corresponde a children
#2 corresponde a Govt_job
#3 corresponde a Private
#4 corresponde a Self-employed
#X[:,6] = [0 if value=="Never_worked" else value for value in X[:,6]]
#X[:,6] = [1 if value=="children" else value for value in X[:,6]]
#X[:,6] = [2 if value=="Govt_job" else value for value in X[:,6]]
#X[:,6] = [3 if value=="Private" else value for value in X[:,6]]
#X[:,6] = [4 if value=="Self-employed" else value for value in X[:,6]]
X[:,5] = [0 if value=="Never_worked" else value for value in X[:,5]]
X[:,5] = [1 if value=="children" else value for value in X[:,5]]
X[:,5] = [2 if value=="Govt_job" else value for value in X[:,5]]
X[:,5] = [3 if value=="Private" else value for value in X[:,5]]
X[:,5] = [4 if value=="Self-employed" else value for value in X[:,5]]

#Con relacion a el lugar en que vive
#0 corresponde a Urban
#1 corresponde a Rural
#X[:,7] = [0 if value=="Urban" else value for value in X[:,7]]
#X[:,7] = [1 if value=="Rural" else value for value in X[:,7]]
X[:,6] = [0 if value=="Urban" else value for value in X[:,6]]
X[:,6] = [1 if value=="Rural" else value for value in X[:,6]]

#Con relacion su estado de fumador
#0 corresponde a never smoked
#1 corresponde a smokes
#2 corresponde a unknowns
#3 corresponde a formerly smoked
#X[:,10] = [0 if value=="never smoked" else value for value in X[:,10]]
#X[:,10] = [1 if value=="smokes" else value for value in X[:,10]]
#X[:,10] = [2 if value=="unknowns" else value for value in X[:,10]]
#X[:,10] = [3 if value=="formerly smoked" else value for value in X[:,10]]
X[:,9] = [0 if value=="never smoked" else value for value in X[:,9]]
X[:,9] = [1 if value=="smokes" else value for value in X[:,9]]
X[:,9] = [2 if value=="unknowns" else value for value in X[:,9]]
X[:,9] = [3 if value=="formerly smoked" else value for value in X[:,9]]

#pase los datos con "," a "." para que python haga lectura de los datos.
#X[:,8]  = [float(s.replace(',', '.')) for s in X[:,8]]
#print (X[:,6])
X[:,7]  = [float(s.replace(',', '.')) for s in X[:,7]]
print (X[:,5])

# Normalización de datos
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print(X_scale)

# Guardar el objeto min_max_scaler
joblib.dump(min_max_scaler, 'min_max_scaler.pkl')
min_max_scaler.data_max_
for num in min_max_scaler.data_max_:
    print("{:8.2f}".format(num), sep=",", end="|")
print()    

for num in min_max_scaler.data_min_:
    print("{:8.2f}".format(num), sep=",", end="|")
print()

#Separacion de datos de entrenamiento y validación
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.1)
print("el largo de X_train es", np.shape(X_train), "y el largo de X_test es", np.shape(X_test))

# Define el criterio de parada temprana
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Construyendo la red
model_2 = Sequential([
    Dense(100, activation='tanh', input_shape=(10,)),
    Dense(1000, activation='tanh'),
    Dense(1000, activation='tanh'),
    Dense(1000, activation='tanh'),
    Dense(1000, activation='tanh'),
    Dense(1000, activation='tanh'),
    Dense(100, activation='tanh'),
    Dense(1, activation='sigmoid'),
])



sgd = optimizers.SGD(learning_rate=0.0001)
model_2.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
#model_2.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Y_train.shape
X = X.astype(float)
Y = Y.astype(float)

#Entrenamiento
X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
X_test = X_test.astype('float32')
Y_test = Y_test.astype('float32')
print(type(X_train[0][0]), type(Y_train[20]))
hist = model_2.fit(X_train, Y_train, batch_size=32, epochs=5000, validation_data=(X_test, Y_test))
model_2.save('my_model.h5')

# Visualización de costo
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Función de desempeño')
plt.ylabel('Costo')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
plt.grid(True)
plt.show()

# Visualización de precisión
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Precisión (accuracy)')
plt.ylabel('Precisión')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
plt.grid(True)
plt.show()
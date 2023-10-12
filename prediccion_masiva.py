from keras.models import load_model 
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo y MinMaxScaler
model = load_model('my_model.h5')
min_max_scaler = joblib.load('min_max_scaler.pkl')

# Carga el archivo csv
#df = pd.read_csv('C:/Users/ricar/Google Drive/Proyecto de titulo Ana Javiera/PT/dataset1.1.csv', sep=";", nrows=2000)
df = pd.read_csv('C:/Users/ricar/Google Drive/Proyecto de titulo Ana Javiera/PT/dataset1.1.csv', sep=";")

# Elimina la primera columna
df = df.drop(df.columns[0], axis=1)

df['Prediction'] = None
df['PredictionS'] = None
predictions_list = []

# Itera sobre cada fila en df
for index, row in df.iterrows():

    # Supongamos que recibimos 10 entradas de algún tipo de interfaz de usuario
    inputs = [row[i] for i in range(0, 10)]  # Asume que los datos están en el mismo orden que en tu código original

    # Convierte categorías a números
    inputs[0] = 0 if inputs[0]=="Male" else 1 if inputs[0]=="Female" else 2  # Gender
    inputs[3] = 0 if inputs[3]==0 else 1  # Heart disease
    inputs[4] = 0 if inputs[4]=="No" else 1  # Ever_married
    inputs[5] = 0 if inputs[5]=="Never_worked" else 1 if inputs[5]=="children" else 2 if inputs[5]=="Govt_job" else 3 if inputs[5]=="Private" else 4  # Work_type
    inputs[6] = 0 if inputs[6]=="Urban" else 1  # Residence_type
    inputs[9] = 0 if inputs[9]=="never smoked" else 1 if inputs[9]=="smokes" else 2 if inputs[9]=="unknowns" else 3  # Smoking_status
    
    inputs[7] = float(inputs[7].replace(',', '.')) if isinstance(inputs[7], str) else inputs[7]  # avg_glucose_level
    inputs[8] = float(inputs[8].replace(',', '.')) if isinstance(inputs[8], str) else inputs[8]  # bmi


    # Asume que las entradas son unidimensionales
    inputs = np.array(inputs).reshape(1, -1)

    # Normalizar las entradas
    inputs = min_max_scaler.transform(inputs)

    # Hacer una predicción con el modelo
    predictions = model.predict(inputs)
    print(predictions[0][0])
    # Añade la predicción a la fila original
    predictions_list.append(predictions[0][0])

# Escribe el DataFrame a un nuevo archivo csv
df['Prediction'] = predictions_list
df['PredictionS'] = df['Prediction'].apply(lambda x: str(x).replace('.', ','))

df.to_csv('dataset_with_predictions_full.csv', sep=';', index=False)

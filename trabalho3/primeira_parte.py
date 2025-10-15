import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import pickle
from sklearn.preprocessing import LabelEncoder


dataset_risco_credito = pd.read_csv('dataset_risco_credito.csv')
print(dataset_risco_credito)

X_risco_credito = dataset_risco_credito.iloc[:, 0:-1].values  # todas as colunas menos a última
y_risco_credito = dataset_risco_credito.iloc[:, -1].values    # apenas a última coluna


X_risco_credito_encoded = X_risco_credito.copy()

label_encoders = []
for i in range(X_risco_credito_encoded.shape[1]):
    le = LabelEncoder()
    X_risco_credito_encoded[:, i] = le.fit_transform(X_risco_credito_encoded[:, i])
    label_encoders.append(le)

le_y = LabelEncoder()
y_risco_credito_encoded = le_y.fit_transform(y_risco_credito)

# Exiba o resultado
print("X_risco_credito codificado:\n", X_risco_credito_encoded)
print("y_risco_credito codificado:\n", y_risco_credito_encoded)



with open('risco_credito.pkl', 'wb') as f:
  pickle.dump([X_risco_credito_encoded, y_risco_credito_encoded], f)


naiveb_risco_credito = GaussianNB()
naiveb_risco_credito.fit(X_risco_credito_encoded, y_risco_credito_encoded)

# Exemplo i: história boa, dívida alta, garantia nenhuma, renda acima_35
# Exemplo ii: história ruim, dívida alta, garantia adequada, renda 0_15

exemplo1 = [
    label_encoders[0].transform(['boa'])[0],
    label_encoders[1].transform(['alta'])[0],
    label_encoders[2].transform(['nenhuma'])[0],
    label_encoders[3].transform(['acima_35'])[0]
]
exemplo2 = [
    label_encoders[0].transform(['ruim'])[0],
    label_encoders[1].transform(['alta'])[0],
    label_encoders[2].transform(['adequada'])[0],
    label_encoders[3].transform(['0_15'])[0]
]

pred1 = naiveb_risco_credito.predict([exemplo1])[0]
pred2 = naiveb_risco_credito.predict([exemplo2])[0]

classe1 = le_y.inverse_transform([pred1])[0]
classe2 = le_y.inverse_transform([pred2])[0]

print('Previsão exemplo i:', classe1)
print('Previsão exemplo ii:', classe2)


print('Classes utilizadas pelo algoritmo:', le_y.inverse_transform(naiveb_risco_credito.classes_))

print('Contagem de registros em cada classe:', dict(zip(le_y.inverse_transform(naiveb_risco_credito.classes_), naiveb_risco_credito.class_count_)))

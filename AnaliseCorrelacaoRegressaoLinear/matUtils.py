import csv
import math
import os

import numpy as np
from numpy.ma.extras import polyfit

import graficos


def regressao_linear(x: list[float], y: list[float]) -> float:
    avg_x = sum(x) / len(x)
    avg_y = sum(y) / len(y)

    dividendo = sum((x[i] - avg_x) * (y[i] - avg_y) for i in range(len(x)))
    divisor = math.sqrt(
        sum((x[i] - avg_x) ** 2 for i in range(len(x))) *
        sum((y[i] - avg_y) ** 2 for i in range(len(y)))
    )

    return dividendo/divisor

def coeficiente_angular(x: list[float], y: list[float]) -> float:
    avg_x = sum(x) / len(x)
    avg_y = sum(y) / len(y)

    dividendo = sum((x[i] - avg_x) * (y[i] - avg_y) for i in range(len(x)))
    divisor = sum((x[i] - avg_x) ** 2 for i in range(len(x)))

    return dividendo / divisor

def get_float_array_from_file(file_path: str, variable_name: str) -> list[float]:
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith(f"{variable_name} ="):
                values_str = line.split('=')[1].strip().strip(';[]')
                return [float(val) for val in values_str.split(';')]
    raise ValueError(f"Variable '{variable_name}' not found in file.")

def ponto_intercepcao(x: list[float], y: list[float]) -> float:
    b1 = coeficiente_angular(x, y)
    avg_y = sum(y) / len(y)
    avg_x = sum(x) / len(x)
    return avg_y - (b1 * avg_x)

def regressao(x: list[float], y: list[float]) -> tuple[float, float]:
    b1 = coeficiente_angular(x, y)
    b0 = ponto_intercepcao(x, y)
    return b0, b1

def correlacao(x: list[float], y: list[float]) -> float:
    return regressao_linear(x, y)

def representacao(x: list[float], y: list[float], i) -> None:
    print(i +": " + str(correlacao(x, y)))
    B0, B1 = regressao(x, y)
    print("B0: " + str(B0))
    print("B1: " + str(B1))
    graficos.plot_dataset(x,y, i)
    print("--------------------------------------------------")


def regressao_multipla_libs():
    x, y = load_data()
    X = np.column_stack((np.ones(len(x)), np.array(x)))
    XT = X.T
    XTX = np.dot(XT, X)
    XTy = np.dot(XT, y)
    beta = np.dot(np.linalg.inv(XTX), XTy)
    return tuple(beta[:3])  # b0, b1, b2

def regressao_multipla(use_scaling=False, scale_y=1e4, feature_scales=(100.0, 10.0)):
    # pega dados do seu próprio load_data()
    x, y = load_data()


    # Escalas para reproduzir ~ (8.95, 1.39, -8.73)
    if use_scaling:
        x = [[row[0]/feature_scales[0], row[1]/feature_scales[1]] for row in x]
        y = [val/scale_y for val in y]

    # garante floats e adiciona intercepto
    X = [[1.0] + [float(v) for v in row] for row in x]
    y = [float(v) for v in y]

    # transposta (único uso de np aqui, além da inverse)
    XT = np.transpose(np.array(X, dtype=float)).tolist()

    def multiplicaMatriz(A, B):
        n, m, p = len(A), len(A[0]), len(B[0])
        return [[sum(A[i][k] * B[k][j] for k in range(m))
                 for j in range(p)] for i in range(n)]

    def matrizVezesVetor(A, v):
        n, m = len(A), len(A[0])
        return [sum(A[i][k] * v[k] for k in range(m)) for i in range(n)]

    # X^T X e X^T y (manuais)
    XTX = multiplicaMatriz(XT, X)
    XTy = matrizVezesVetor(XT, y)

    #INVERSE
    XTX_inv = np.linalg.inv(np.array(XTX, dtype=float)).tolist()

    # beta = (X^T X)^-1 (X^T y) (manual)
    beta = matrizVezesVetor(XTX_inv, XTy)

    # retorna apenas b0, b1, b2
    b0, b1, b2 = beta[:3]
    return b0, b1, b2


def load_data():
    data = [
        [2104, 3, 3.999e+05],
        [1600, 3, 3.299e+05],
        [2400, 3, 3.690e+05],
        [1416, 2, 2.320e+05],
        [3000, 4, 5.399e+05],
        [1985, 4, 2.999e+05],
        [1534, 3, 3.149e+05],
        [1427, 3, 1.990e+05],
        [1380, 3, 2.120e+05],
        [1494, 3, 2.425e+05],
        [1940, 4, 2.400e+05],
        [2000, 3, 3.470e+05],
        [1890, 3, 3.300e+05],
        [4478, 5, 6.999e+05],
        [1268, 3, 2.599e+05],
        [2300, 4, 4.499e+05],
        [1320, 2, 2.999e+05],
        [1236, 3, 1.999e+05],
        [2609, 4, 5.000e+05],
        [3031, 4, 5.990e+05],
        [1767, 3, 2.529e+05],
        [1888, 2, 2.550e+05],
        [1604, 3, 2.429e+05],
        [1962, 4, 2.599e+05],
        [3890, 3, 5.739e+05],
        [1100, 3, 2.499e+05],
        [1458, 3, 4.645e+05],
        [2526, 3, 4.690e+05],
        [2200, 3, 4.750e+05],
        [2637, 3, 2.999e+05],
        [1839, 2, 3.499e+05],
        [1000, 1, 1.699e+05],
        [2040, 4, 3.149e+05],
        [3137, 3, 5.799e+05],
        [1811, 4, 2.859e+05],
        [1437, 3, 2.499e+05],
        [1239, 3, 2.299e+05],
        [2132, 4, 3.450e+05],
        [4215, 4, 5.490e+05],
        [2162, 4, 2.870e+05],
        [1664, 2, 3.685e+05],
        [2238, 3, 3.299e+05],
        [2567, 4, 3.140e+05],
        [1200, 3, 2.990e+05],
        [852, 2, 1.799e+05],
        [1852, 4, 2.999e+05],
        [1203, 3, 2.395e+05],
    ]

    data = np.array(data, dtype=float)

    # separa variáveis independentes (x) e dependente (y)
    x = data[:, :2].tolist()
    y = data[:, 2].tolist()

    return x, y

def load_data_tamanho_casa_preco():
    data = [
        [2104, 3, 3.999e+05],
        [1600, 3, 3.299e+05],
        [2400, 3, 3.690e+05],
        [1416, 2, 2.320e+05],
        [3000, 4, 5.399e+05],
        [1985, 4, 2.999e+05],
        [1534, 3, 3.149e+05],
        [1427, 3, 1.990e+05],
        [1380, 3, 2.120e+05],
        [1494, 3, 2.425e+05],
        [1940, 4, 2.400e+05],
        [2000, 3, 3.470e+05],
        [1890, 3, 3.300e+05],
        [4478, 5, 6.999e+05],
        [1268, 3, 2.599e+05],
        [2300, 4, 4.499e+05],
        [1320, 2, 2.999e+05],
        [1236, 3, 1.999e+05],
        [2609, 4, 5.000e+05],
        [3031, 4, 5.990e+05],
        [1767, 3, 2.529e+05],
        [1888, 2, 2.550e+05],
        [1604, 3, 2.429e+05],
        [1962, 4, 2.599e+05],
        [3890, 3, 5.739e+05],
        [1100, 3, 2.499e+05],
        [1458, 3, 4.645e+05],
        [2526, 3, 4.690e+05],
        [2200, 3, 4.750e+05],
        [2637, 3, 2.999e+05],
        [1839, 2, 3.499e+05],
        [1000, 1, 1.699e+05],
        [2040, 4, 3.149e+05],
        [3137, 3, 5.799e+05],
        [1811, 4, 2.859e+05],
        [1437, 3, 2.499e+05],
        [1239, 3, 2.299e+05],
        [2132, 4, 3.450e+05],
        [4215, 4, 5.490e+05],
        [2162, 4, 2.870e+05],
        [1664, 2, 3.685e+05],
        [2238, 3, 3.299e+05],
        [2567, 4, 3.140e+05],
        [1200, 3, 2.990e+05],
        [852, 2, 1.799e+05],
        [1852, 4, 2.999e+05],
        [1203, 3, 2.395e+05],
    ]
    data = np.array(data, dtype=int)
    X = data[:, 0].tolist()  # tamanho da casa
    Y = data[:, 2].tolist()  # preço
    return X, Y

def load_data_quartos_preco():
    data = [
        [2104, 3, 3.999e+05],
        [1600, 3, 3.299e+05],
        [2400, 3, 3.690e+05],
        [1416, 2, 2.320e+05],
        [3000, 4, 5.399e+05],
        [1985, 4, 2.999e+05],
        [1534, 3, 3.149e+05],
        [1427, 3, 1.990e+05],
        [1380, 3, 2.120e+05],
        [1494, 3, 2.425e+05],
        [1940, 4, 2.400e+05],
        [2000, 3, 3.470e+05],
        [1890, 3, 3.300e+05],
        [4478, 5, 6.999e+05],
        [1268, 3, 2.599e+05],
        [2300, 4, 4.499e+05],
        [1320, 2, 2.999e+05],
        [1236, 3, 1.999e+05],
        [2609, 4, 5.000e+05],
        [3031, 4, 5.990e+05],
        [1767, 3, 2.529e+05],
        [1888, 2, 2.550e+05],
        [1604, 3, 2.429e+05],
        [1962, 4, 2.599e+05],
        [3890, 3, 5.739e+05],
        [1100, 3, 2.499e+05],
        [1458, 3, 4.645e+05],
        [2526, 3, 4.690e+05],
        [2200, 3, 4.750e+05],
        [2637, 3, 2.999e+05],
        [1839, 2, 3.499e+05],
        [1000, 1, 1.699e+05],
        [2040, 4, 3.149e+05],
        [3137, 3, 5.799e+05],
        [1811, 4, 2.859e+05],
        [1437, 3, 2.499e+05],
        [1239, 3, 2.299e+05],
        [2132, 4, 3.450e+05],
        [4215, 4, 5.490e+05],
        [2162, 4, 2.870e+05],
        [1664, 2, 3.685e+05],
        [2238, 3, 3.299e+05],
        [2567, 4, 3.140e+05],
        [1200, 3, 2.990e+05],
        [852, 2, 1.799e+05],
        [1852, 4, 2.999e+05],
        [1203, 3, 2.395e+05],
    ]
    data = np.array(data, dtype=int)
    Z = data[:, 1].tolist()  # número de quartos
    V = data[:, 2].tolist()  # preço
    return Z, V


def load_data_fase3():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'datafase3.csv')
    data = []
    with open(csv_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                data.append([float(parts[0]), float(parts[1])])

    data = np.array(data, dtype=float)
    x = data[:, 0].tolist()  # primeira coluna
    y = data[:, 1].tolist()  # segunda coluna
    return x, y


def demo_regressaop(x,y,n):
    return polyfit(x,y,n);

def EQM_manual(y, yhat):
    n = len(y)
    soma = sum((y[i] - yhat[i]) ** 2 for i in range(n))
    return soma / n

def getYhatManual(x, bN):
    x = np.asarray(x, dtype=float)
    y_chapeu = np.zeros_like(x, dtype=float)
    for i, coef in enumerate(bN):
        y_chapeu += coef * (x ** i)   # usa os x originais, não linspace
    return y_chapeu

def calcular_eqm_multiplos_graus(x, y, graus):
    resultados = {}

    for grau in graus:
        bN = demo_regressaop(x, y, grau)
        bN_invertida = bN[::-1]
        yHat = getYhatManual(x, bN_invertida)
        eqm = EQM_manual(y, yHat)
        resultados[grau] = eqm
        print(f"EQM grau {grau}: {eqm}")

    return resultados
import csv
import math

import numpy as np

import demo


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
    demo.plot_dataset(x,y, i)
    print("--------------------------------------------------")
def regressao_multipla():
    x, y = load_data()  # espere que x seja lista de listas e y lista (1D)

    # Garante tipos float e adiciona intercepto
    X = [[1.0] + [float(v) for v in row] for row in x]
    y = [float(v) for v in y]

    # Transposta com NumPy (único uso do np), depois converte pra lista pura
    XT = np.transpose(np.array(X, dtype=float)).tolist()

    # ----------------- utilitários sem numpy -----------------
    def matmul(A, B):
        n, m, p = len(A), len(A[0]), len(B[0])
        # A: n x m, B: m x p
        return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

    def matvec(A, v):
        n, m = len(A), len(A[0])
        return [sum(A[i][k] * v[k] for k in range(m)) for i in range(n)]

    def inverse(M):
        # Gauss-Jordan com pivotamento parcial
        n = len(M)
        A = [row[:] for row in M]
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for col in range(n):
            # escolhe pivô (maior valor absoluto na coluna 'col' das linhas col..n-1)
            piv = max(range(col, n), key=lambda r: abs(A[r][col]))
            if abs(A[piv][col]) < 1e-12:
                raise ValueError("Matriz (X^T X) é singular ou quase singular.")

            # troca linhas se necessário
            if piv != col:
                A[col], A[piv] = A[piv], A[col]
                I[col], I[piv] = I[piv], I[col]

            # normaliza linha do pivô
            factor = A[col][col]
            for j in range(n):
                A[col][j] /= factor
                I[col][j] /= factor

            # zera as outras linhas na coluna 'col'
            for i in range(n):
                if i == col:
                    continue
                f = A[i][col]
                if f != 0.0:
                    for j in range(n):
                        A[i][j] -= f * A[col][j]
                        I[i][j] -= f * I[col][j]

        return I
    # ---------------------------------------------------------

    # X^T X e X^T y
    XTX = matmul(XT, X)
    XTy = matvec(XT, y)

    # (X^T X)^-1
    XTX_inv = inverse(XTX)

    # beta = (X^T X)^-1 (X^T y)
    beta = matvec(XTX_inv, XTy)

    # Desempacota os três primeiros coeficientes
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
    x = data[:, :2]
    y = data[:, 2]
    return x, y

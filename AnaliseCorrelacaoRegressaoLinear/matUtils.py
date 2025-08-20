import math


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

def print_statements(x: list[float], y: list[float],i) -> None:
    print(i +": " + str(correlacao(x, y)))
    B1, B2 = regressao(x, y)
    print("B1: " + str(B1))
    print("B2: " + str(B2))
    print("--------------------------------------------------")
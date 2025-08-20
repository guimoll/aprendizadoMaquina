import matUtils
import matplotlib.pyplot as plt


def plot_dataset(x, y, name):
    correlacao = matUtils.correlacao(x, y)
    intercept, slope = matUtils.regressao(x, y)
    line = [slope * xi + intercept for xi in x]

    plt.scatter(x, y, label='Dados')
    plt.plot(x, line, color='red', label='Regressão')
    plt.title(f"{name}\nCorrelação: {correlacao:.5f}, Regressão: y={slope:.5f}x+{intercept:.5f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
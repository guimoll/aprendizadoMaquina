import matUtils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matUtils
import numpy as np
import plotly.graph_objects as go

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

def plot_regression_3d(b0, b1, b2):


    # seus dados
    x, y = matUtils.load_data()
    x1 = np.array([row[0] for row in x], dtype=float)  # tamanho
    x2 = np.array([row[1] for row in x], dtype=float)  # quartos
    y  = np.array(y, dtype=float)                      # preço

    # grade para o plano de regressão
    x1_lin = np.linspace(x1.min(), x1.max(), 30)
    x2_lin = np.linspace(x2.min(), x2.max(), 30)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    Yhat = b0 + b1 * X1 + b2 * X2

    # pontos e plano
    scatter = go.Scatter3d(
        x=x1, y=x2, z=y,
        mode="markers",
        name="Dados",
        marker=dict(size=4, opacity=0.85)
    )
    surface = go.Surface(
        x=X1, y=X2, z=Yhat,
        name="Plano de regressão",
        opacity=0.5,
        showscale=False
    )

    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        title="Dispersão 3D + Plano de Regressão",
        scene=dict(
            xaxis_title="Tamanho da casa",
            yaxis_title="Nº de quartos",
            zaxis_title="Preço"
        ),
        legend=dict(y=0.95, x=0.01)
    )

    fig.show()
    return fig
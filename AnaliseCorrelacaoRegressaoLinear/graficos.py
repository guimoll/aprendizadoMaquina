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

    # Regression line (red) along x1, with x2 fixed at its mean
    x2_mean = np.mean(x2)
    line_x1 = np.linspace(x1.min(), x1.max(), 100)
    line_y = b0 + b1 * line_x1 + b2 * x2_mean

    fig.show()
    return fig


def plot_dataset_regressao_3d(x, y, name: str):
    r = matUtils.correlacao(x, y)
    b0, b1 = matUtils.regressao(x, y)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # linha de regressão (nos extremos do x)
    x_lin = np.linspace(x.min(), x.max(), 100)
    y_hat = b0 + b1 * x_lin

    # pontos (3D com eixo "dummy" para manter padrão 3D)
    scatter = go.Scatter3d(
        x=x,
        y=np.zeros_like(x),    # eixo dummy para visual 3D
        z=y,
        mode="markers",
        name="Dados",
        marker=dict(size=4, opacity=0.85)
    )

    # linha da regressão
    line = go.Scatter3d(
        x=x_lin,
        y=np.zeros_like(x_lin),
        z=y_hat,
        mode="lines",
        name=f"Regressão",
        line=dict(width=5)
    )

    fig = go.Figure(data=[line, scatter])
    fig.update_layout(
        title=f"{name} — r={r:.5f} | y = {b1:.5f}·x + {b0:.5f}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="",       # dummy
            zaxis_title="Y"
        ),
        legend=dict(y=0.95, x=0.01),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # anotação fixa no canto superior (papel), deixando o r explícito na figura
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"Correlação (r): {r:.5f}",
        showarrow=False,
        font=dict(size=12)
    )

    fig.show()
    return fig

def plot_dataset_regressao_Fase3_itemD(x, y, name: str,b0,b1):
    r = matUtils.correlacao(x, y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # linha de regressão (nos extremos do x)
    x_lin = np.linspace(x.min(), x.max(), 100)
    y_hat = b0 + b1 * x_lin

    # pontos (3D com eixo "dummy" para manter padrão 3D)
    scatter = go.Scatter3d(
        x=x,
        y=np.zeros_like(x),    # eixo dummy para visual 3D
        z=y,
        mode="markers",
        name="Dados",
        marker=dict(size=4, opacity=0.85)
    )

    # linha da regressão
    line = go.Scatter3d(
        x=x_lin,
        y=np.zeros_like(x_lin),
        z=y_hat,
        mode="lines",
        name=f"Regressão",
        line=dict(width=5, color='red'),
    )

    fig = go.Figure(data=[line, scatter])
    fig.update_layout(
        title=f"{name} — r={r:.5f} | y = {b1:.5f}·x + {b0:.5f}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="",       # dummy
            zaxis_title="Y"
        ),
        legend=dict(y=0.95, x=0.01),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # anotação fixa no canto superior (papel), deixando o r explícito na figura
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"Correlação (r): {r:.5f}",
        showarrow=False,
        font=dict(size=12)
    )

    fig.show()
    return fig

def plot_regression_3d_itemD(x,y,b0, b1, b2):
    # seus dados
    x1 = np.array([row[0] for row in x], dtype=float)
    x2 = np.array([row[1] for row in x], dtype=float)
    y  = np.array(y, dtype=float)

    # grade para o plano de regressão
    x1_lin = np.linspace(x1.min(), x1.max(), 30)
    x2_lin = np.linspace(x2.min(), x2.max(), 30)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    Yhat = b0 +(b1 * X1) + (b2 * (X2)**2)

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

    # Regression line (red) along x1, with x2 fixed at its mean
    x2_mean = np.mean(x2)
    line_x1 = np.linspace(x1.min(), x1.max(), 100)
    line_y = b0 + b1 * line_x1 + b2 * x2_mean

    fig.show()
    return fig

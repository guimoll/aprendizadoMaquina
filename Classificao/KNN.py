import scipy.io as scipy
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

def dist(x,y):
    return sum((a-b)**2 for a,b in zip(x,y))**0.5

# matriz de 50 por 100
#Fazer matriz de distancia. Para cada exemplo de teste, calcule a distancia com cada ex. de treinamento e salve
# na matriz de distancia

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k, padronizacao=False):
    # Conversão para numpy
    dadosTrain_np = np.asarray(dadosTrain, dtype=float)
    dadosTeste_np = np.asarray(dadosTeste, dtype=float)

    #Normaliza os dados, ajustando-os / media e desvio padrao
    if padronizacao:
        mean = np.mean(dadosTrain_np, axis=0, keepdims=True)
        std  = np.std(dadosTrain_np, axis=0, keepdims=True) + 1e-8
        dadosTrain_proc = (dadosTrain_np - mean) / std
        dadosTeste_proc = (dadosTeste_np  - mean) / std
    else:
        dadosTrain_proc = dadosTrain_np
        dadosTeste_proc = dadosTeste_np

    matrizDistancia = []
    for exemploTeste in dadosTeste_proc:
        linhaDistancia = []
        for exemploTrain in dadosTrain_proc:
            linhaDistancia.append(dist(exemploTeste, exemploTrain))
        matrizDistancia.append(linhaDistancia)

    rotulosPrevistos = []
    #percorre para prever o rotulo de cada amostra de teste
    for i in range(len(dadosTeste_proc)):
        distancias = matrizDistancia[i]
        indicesKMenores = np.argsort(distancias)[:k]
        rotulosKMenores = [int(rotuloTrain[j]) for j in indicesKMenores]
        rotuloPrevisto = mode(rotulosKMenores, keepdims=False).mode
        rotulosPrevistos.append(rotuloPrevisto)

    return rotulosPrevistos


def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []
    for idx in range(0, len(dados)):
        if(rotulos[idx] == rotulo):
            ret.append(dados[idx][indice])
    return ret

def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()
    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), getDadosRotulo(dados, rotulos, 1, d2), c='red' , marker='^')
    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), getDadosRotulo(dados, rotulos, 2, d2), c='blue' , marker='+')
    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), getDadosRotulo(dados, rotulos, 3, d2), c='green', marker='.')

    plt.show()

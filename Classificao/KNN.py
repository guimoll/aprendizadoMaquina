import scipy.io as scipy
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

def dist(x,y):
    return sum((a-b)**2 for a,b in zip(x,y))**0.5

# matriz de 50 por 100
#Fazer matriz de distancia. Para cada exemplo de teste, calcule a distancia com cada ex. de treinamento e salve
# na matriz de distancia

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    matrizDistancia = []
    for exemploTeste in dadosTeste:
        linhaDistancia = []
        for exemploTrain in dadosTrain:
            linhaDistancia.append(dist(exemploTeste, exemploTrain))
        matrizDistancia.append(linhaDistancia)

    rotulosPrevistos = []
    for i in range(len(dadosTeste)):
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

if __name__ == '__main__':

    mat = scipy.loadmat('data/grupoDados1.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots'].flatten() #flatten pra ser uma matriz de 1 dimensão
    trainRots = mat['trainRots'].flatten()  # flatten pra ser uma matriz de 1 dimensão

    rotuloPrevisto = np.array(meuKnn(grupoTrain, trainRots, grupoTest, 10))
    print("Predicted labels:", rotuloPrevisto)

    estaCorreto = rotuloPrevisto == testRots
    print("Correct predictions (boolean array):", estaCorreto)

    numCorreto = np.sum(estaCorreto)
    print("Number of correct predictions:", numCorreto)

    totalNum = len(testRots)
    print("Total number of test samples:", totalNum)

    acuracia = numCorreto / totalNum
    print("Accuracy:", acuracia)

    visualizaPontos(grupoTrain, trainRots, 0, 1)


best_acc = 0
best_k = 1
for k in range(1, 49):
    rotuloPrevisto = np.array(meuKnn(grupoTrain, trainRots, grupoTest, k))
    acuracia = np.sum(rotuloPrevisto == testRots) / len(testRots)
    if acuracia > best_acc:
        best_acc = acuracia
        best_k = k

print("Melhor acuracia:", best_acc)
print("Melhor numero de vizinhos:", best_k)
#distanca euclidiana nos slides tem que implementar na mao
import scipy.io as scipy
import numpy as np


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
        indicesKMenores = sorted(range(len(distancias)), key=lambda x: distancias[x])[:k]
        # conversão pro int
        rotulosKMenores = [int(rotuloTrain[j]) for j in indicesKMenores]
        rotuloPrevisto = max(set(rotulosKMenores), key=rotulosKMenores.count)
        rotulosPrevistos.append(rotuloPrevisto)

    return rotulosPrevistos



if __name__ == '__main__':

    mat = scipy.loadmat('data/grupoDados1.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots'].flatten()  # Ensure 1D
    trainRots = mat['trainRots'].flatten()  # Ensure 1D

    rotuloPrevisto = np.array(meuKnn(grupoTrain, trainRots, grupoTest, 1))
    print("Predicted labels:", rotuloPrevisto)

    estaCorreto = rotuloPrevisto == testRots
    print("Correct predictions (boolean array):", estaCorreto)

    numCorreto = np.sum(estaCorreto)
    print("Number of correct predictions:", numCorreto)

    totalNum = len(testRots)
    print("Total number of test samples:", totalNum)

    acuracia = numCorreto / totalNum
    print("Accuracy:", acuracia)

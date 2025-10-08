import KNN
import numpy as np


if __name__ == '__main__':

    mat = KNN.scipy.loadmat('data/grupoDados4.mat')
    testSet = mat['testSet']
    trainSet = mat['trainSet']
    testLabs = mat['testLabs'].flatten() #flatten pra ser uma matriz de 1 dimensão
    trainLabs = mat['trainLabs'].flatten()  # flatten pra ser uma matriz de 1 dimensão

    rotuloPrevisto = np.array(KNN.meuKnn(trainSet, trainLabs, testSet, 9, True))
    print("Predicted labels:", rotuloPrevisto)

    estaCorreto = rotuloPrevisto == testLabs
    print("Correct predictions (boolean array):", estaCorreto)

    numCorreto = np.sum(estaCorreto)
    print("Number of correct predictions:", numCorreto)

    totalNum = len(testLabs)
    print("Total number of test samples:", totalNum)

    acuracia = numCorreto / totalNum
    print("Accuracy:", acuracia)

    KNN.visualizaPontos(trainSet, trainLabs, 0, 1)


best_acc = 0
best_k = 1
k_92 = 0
for k in range(1, 49):
    rotuloPrevisto = KNN.np.array(KNN.meuKnn(trainSet, trainLabs, testSet, k, True))
    acuracia = np.sum(rotuloPrevisto == testLabs) / len(testLabs)
    if acuracia > best_acc:
        best_acc = acuracia
        best_k = k

    # Para dar tolerancia
    if abs(acuracia - 0.92) < 0.005:
        k_92 = k

print("Melhor acuracia:", best_acc)
print("Melhor numero de vizinhos:", best_k)
print("Acuracia 92% atingida com k =", k_92)


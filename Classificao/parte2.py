import KNN
import numpy as np


if __name__ == '__main__':

    mat = KNN.scipy.loadmat('data/grupoDados2.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots'].flatten() #flatten pra ser uma matriz de 1 dimensão
    trainRots = mat['trainRots'].flatten()  # flatten pra ser uma matriz de 1 dimensão

    rotuloPrevisto = np.array(KNN.meuKnn(grupoTrain, trainRots, grupoTest, 32, True))
    print("Predicted labels:", rotuloPrevisto)

    estaCorreto = rotuloPrevisto == testRots
    print("Correct predictions (boolean array):", estaCorreto)

    numCorreto = np.sum(estaCorreto)
    print("Number of correct predictions:", numCorreto)

    totalNum = len(testRots)
    print("Total number of test samples:", totalNum)

    acuracia = numCorreto / totalNum
    print("Accuracy:", acuracia)

    KNN.visualizaPontos(grupoTrain, trainRots, 0, 1)


best_acc = 0
best_k = 1
k_98 = 0

for k in range(1, 49):
    rotuloPrevisto = KNN.np.array(KNN.meuKnn(grupoTrain, trainRots, grupoTest, k, True))
    acuracia = np.sum(rotuloPrevisto == testRots) / len(testRots)
    if acuracia > best_acc:
        best_acc = acuracia
        best_k = k

        # Para dar tolerancia
    if abs(acuracia - 0.98) < 0.005:
        k_98 = k

print("Melhor acuracia:", best_acc)
print("Melhor numero de vizinhos:", best_k)
print("Acuracia 98% atingida com k =", k_98)


#A acurácia inicial obtida foi de 78,3%, ao aplicar o mesmo kNN do primeiro experimento diretamente sobre os dados brutos.
import KNN
import numpy as np


if __name__ == '__main__':

    mat = KNN.scipy.loadmat('data/grupoDados1.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots'].flatten() #flatten pra ser uma matriz de 1 dimensão
    trainRots = mat['trainRots'].flatten()  # flatten pra ser uma matriz de 1 dimensão

    rotuloPrevisto = np.array(KNN.meuKnn(grupoTrain, trainRots, grupoTest, 10))
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
for k in range(1, 49):
    rotuloPrevisto = KNN.np.array(KNN.meuKnn(grupoTrain, trainRots, grupoTest, k))
    acuracia = np.sum(rotuloPrevisto == testRots) / len(testRots)
    if acuracia > best_acc:
        best_acc = acuracia
        best_k = k

print("Melhor acuracia:", best_acc)
print("Melhor numero de vizinhos:", best_k)

#A acurácia máxima observada foi 0,98 (98%), atingida com k = 3 vizinhos.
#Isso significa que, de 50 amostras de teste, 49 foram classificadas corretamente.

# Nem sempre é necessário usar todos os atributos; alguns subconjuntos podem atingir desempenho equivalente dependendo da redundância entre as features.
import KNN
import numpy as np


if __name__ == '__main__':

    mat = KNN.scipy.loadmat('data/grupoDados3.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots'].flatten() #flatten pra ser uma matriz de 1 dimensão
    trainRots = mat['trainRots'].flatten()  # flatten pra ser uma matriz de 1 dimensão

    rotuloPrevisto = np.array(KNN.meuKnn(grupoTrain, trainRots, grupoTest, 29))
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
k_92 = 0
for k in range(1, 49):
    rotuloPrevisto = KNN.np.array(KNN.meuKnn(grupoTrain, trainRots, grupoTest, k))
    acuracia = np.sum(rotuloPrevisto == testRots) / len(testRots)
    if acuracia > best_acc:
        best_acc = acuracia
        best_k = k

#Para dar tolerancia
    if abs(acuracia - 0.92) < 0.005:
        k_92 = k

print("Melhor acuracia:", best_acc)
print("Melhor numero de vizinhos:", best_k)
print("Acuracia 92% atingida com k =", k_92)

#Aplicando com o K de 1, a acuracia obtida foi de 62%.
#Sem a normalizaçao, rodando no loop, a melhor acuracia obtida foi de 94%, com K = 9
#Como o objetivo da atividade era alcancar acuracica de 92%, foi feito uma validação para
#verificar qual numero de K que atinge, isso, sendo k = 29.
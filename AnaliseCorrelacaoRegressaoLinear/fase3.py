import matUtils
import graficos
#ITEM A
#x,y = matUtils.load_data_fase3(False,0)
#print(matUtils.load_data_fase3(False,0))

#ITEM B
#x,y = matUtils.load_data_fase3(False,0)
#graficos.plot_dataset(x,y, "Grafico dispersao item b")


#ITEM C
x,y = matUtils.load_data_fase3(False,0)
bn_train = matUtils.demo_regressaop(x, y, 1)
bn_invertida_train = bn_train[::-1]
graficos.plot_regression_3d_fase3(x, y, "Grafico n1 item c",'red', bn_invertida_train)

#ITEM D
x,y = matUtils.load_data_fase3(False,0)
bn_train = matUtils.demo_regressaop(x, y, 2)
bn_invertida_train = bn_train[::-1]
graficos.plot_regression_3d_fase3(x, y,"Grafico n2 item d",'green', bn_invertida_train)

#ITEM E
x,y = matUtils.load_data_fase3(False,0)
bn_train = matUtils.demo_regressaop(x, y, 3)
bn_invertida_train = bn_train[::-1]
graficos.plot_regression_3d_fase3(x, y,"Grafico n3 item E",'black', bn_invertida_train)

#ITEM F
x,y = matUtils.load_data_fase3(False,0)
bn_train = matUtils.demo_regressaop(x, y, 8)
bn_invertida_train = bn_train[::-1]
graficos.plot_regression_3d_fase3(x, y,"Grafico n8 item F",'yellow', bn_invertida_train)

#ITEM G
x,y = matUtils.load_data_fase3(False,0)
print("EQM manual de varios graus:", matUtils.calcular_eqm_multiplos_graus(x,y,[1,2,3,8]))
print("------------------------------------------------")
print("EQM da base de dados separada para treino")
#Item H,I

x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
bn_train = matUtils.demo_regressaop(x_treino, y_treino, 1)
bn_invertida_train = bn_train[::-1]
graficos.plot_regression_3d_fase3(x_treino, y_treino, "Grafico treinamento n1",'red', bn_invertida_train)

x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
bn_train = matUtils.demo_regressaop(x_treino, y_treino, 2)
bn_invertida_train = bn_train[::-1]
graficos.plot_regression_3d_fase3(x_treino, y_treino,"Grafico treinamento n2",'green', bn_invertida_train)

x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
bn_train = matUtils.demo_regressaop(x_treino, y_treino, 3)
bn_invertida_train = bn_train[::-1]
graficos.plot_regression_3d_fase3(x_treino, y_treino,"Grafico treinamento n3",'black', bn_invertida_train)

x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
bn_train = matUtils.demo_regressaop(x_treino, y_treino, 8)
bn_invertida_train = bn_train[::-1]
graficos.plot_regression_3d_fase3(x_treino, y_treino,"Grafico treinamento n8",'yellow', bn_invertida_train)


x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
bn_train = matUtils.demo_regressaop(x_teste, y_teste, 1)
bn_invertida_teste = bn_train[::-1]
graficos.plot_regression_3d_fase3(x_teste, y_teste, "Grafico teste n1",'red', bn_invertida_teste)

x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
bn_train = matUtils.demo_regressaop(x_teste, y_teste, 2)
bn_invertida_teste = bn_train[::-1]
graficos.plot_regression_3d_fase3(x_teste, y_teste,"Grafico teste n2",'green', bn_invertida_teste)

x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
bn_train = matUtils.demo_regressaop(x_teste, y_teste, 3)
bn_invertida_teste = bn_train[::-1]
graficos.plot_regression_3d_fase3(x_teste, y_teste,"Grafico teste n3",'black', bn_invertida_teste)

x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
bn_train = matUtils.demo_regressaop(x_teste, y_teste, 8)
bn_invertida_teste = bn_train[::-1]
graficos.plot_regression_3d_fase3(x_teste, y_teste,"Grafico teste n8",'yellow', bn_invertida_teste)



#Item J
x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
print(matUtils.calcular_eqm_multiplos_graus(x_teste,y_teste,[1,2,3,8]))

x_train, y_train, x_test, y_test = matUtils.load_data_fase3(True, 0.1)
bn_train = matUtils.demo_regressaop(x_train, y_train, 2)
bn_invertida_train = bn_train[::-1]
bn_test = matUtils.demo_regressaop(x_test, y_test, 2)
bn_invertida_test = bn_test[::-1]


yHat_train = matUtils.getYhatManual(x_train, bn_invertida_train)
yHat_test = matUtils.getYhatManual(x_test, bn_invertida_test)

eqm_train = matUtils.EQM_manual(y_train, yHat_train)
eqm_test = matUtils.EQM_manual(y_test, yHat_test)
r2_train, r2_test = matUtils.calcular_r2(x_train, y_train, x_test, y_test, bn_invertida_train, bn_invertida_test)


print("EQM (treino):", eqm_train)
print("EQM (teste):", eqm_test)
print("R² (treino):", r2_train)
print("R² (teste):", r2_test)

#o modelo é mais preciso nos dados de teste, já que apresenta menor erro quadrático médio e
# maior coeficiente de determinação, indicando boa capacidade de generalização para dados não vistos.

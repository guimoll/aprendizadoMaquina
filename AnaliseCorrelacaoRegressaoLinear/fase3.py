import matUtils
import graficos
#ITEM A
#x,y = matUtils.load_data_fase3(False,0)
#print(matUtils.load_data_fase3(False,0))

#ITEM B
#x,y = matUtils.load_data_fase3(False,0)
#graficos.plot_dataset(x,y, "Grafico dispersao item b")


#ITEM C
#x,y = matUtils.load_data_fase3(False,0)
#bN = matUtils.demo_regressaop(x, y, 1)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x,y, "Grafico n1 item c",'red',bN_invertida)

#ITEM D
#x,y = matUtils.load_data_fase3(False,0)
#bN = matUtils.demo_regressaop(x, y, 2)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x,y,"Grafico n2 item d",'green',bN_invertida)

#ITEM E
#x,y = matUtils.load_data_fase3(False,0)
#bN = matUtils.demo_regressaop(x, y, 3)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x,y,"Grafico n3 item E",'black',bN_invertida)

#ITEM F
#x,y = matUtils.load_data_fase3(False,0)
#bN = matUtils.demo_regressaop(x, y, 8)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x,y,"Grafico n8 item F",'yellow',bN_invertida)

#ITEM G
#x,y = matUtils.load_data_fase3(True,0)
#print("EQM manual de varios graus:", matUtils.calcular_eqm_multiplos_graus(x,y,[1,2,3,8]))

#Item H,I

#x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
#bN = matUtils.demo_regressaop(x_treino, y_treino, 1)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x_treino,y_treino, "Grafico treinamento n1",'red',bN_invertida)

#x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
#bN = matUtils.demo_regressaop(x_treino, y_treino, 2)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x_treino,y_treino,"Grafico treinamento n2",'green',bN_invertida)

#x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
#bN = matUtils.demo_regressaop(x_treino, y_treino, 3)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x_treino,y_treino,"Grafico treinamento n8",'black',bN_invertida)

#x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
#bN = matUtils.demo_regressaop(x_treino, y_treino, 8)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x_treino,y_treino,"Grafico treinamento n8",'yellow',bN_invertida)

#Item J
#x_treino, y_treino, x_teste, y_teste = matUtils.load_data_fase3(True,0.1)
#print(matUtils.calcular_eqm_multiplos_graus(x_teste,y_teste,[1,2,3,8]))

x_train, y_train, x_test, y_test = matUtils.load_data_fase3(True, 0.1)
bN = matUtils.demo_regressaop(x_train, y_train, 2)   # grau 2
bN_invertida = bN[::-1]

r2_train, r2_test = matUtils.calcular_r2(x_train, y_train, x_test, y_test, bN_invertida)

print("R² (treino):", r2_train)
print("R² (teste):", r2_test)
#O modelo explica bem os dados de treino (≈78%), mas cai para ≈53% nos dados de teste, indicando perda de generalização e possível overfitting.

#O modelo mais preciso seria o com o menor coeficiente de erro. TERMINAR ESSE ITEM
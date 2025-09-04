import matUtils
import graficos
#ITEM A
#x,y = matUtils.load_data_fase3()
#print(matUtils.load_data_fase3())

#ITEM B
#x,y = matUtils.load_data_fase3()
#graficos.plot_dataset(x,y, "Grafico dispersao item b")


#ITEM C
#x,y = matUtils.load_data_fase3()
#bN = matUtils.demo_regressaop(x, y, 1)
#bN_invertida = bN[::-1]
#graficos.plot_dataset_regressao_Fase3_itemD(x,y, "Grafico regressao item d",bN_invertida[0], bN_invertida[1])

#ITEM D
x,y = matUtils.load_data_fase3()
bN = matUtils.demo_regressaop(x, y, 2)
bN_invertida = bN[::-1]
graficos.plot_regression_3d_fase3(x,y,bN_invertida[0], bN_invertida[1], bN_invertida[2])
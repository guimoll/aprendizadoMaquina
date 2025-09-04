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
#graficos.plot_regression_3d_fase3(x,y, "Grafico n1 item c",'red',bN_invertida)

#ITEM D
#x,y = matUtils.load_data_fase3()
#bN = matUtils.demo_regressaop(x, y, 2)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x,y,"Grafico n2 item d",'green',bN_invertida)

#ITEM E
#x,y = matUtils.load_data_fase3()
#bN = matUtils.demo_regressaop(x, y, 3)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x,y,"Grafico n3 item E",'black',bN_invertida)

#ITEM E
#x,y = matUtils.load_data_fase3()
#bN = matUtils.demo_regressaop(x, y, 8)
#bN_invertida = bN[::-1]
#graficos.plot_regression_3d_fase3(x,y,"Grafico n8 item F",'yellow',bN_invertida)

#ITEM G
x,y = matUtils.load_data_fase3()
bN = matUtils.demo_regressaop(x, y, 2)
bN_invertida = bN[::-1]
yHat = matUtils.getYhatManual(x, bN_invertida)
print("EQM manual:", matUtils.EQM_manual(y, yHat))
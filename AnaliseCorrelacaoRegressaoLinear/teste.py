# Guilherme Moll, Gabriel Ramos, Eduardo brandt
# o dataset R3 não é apropriado pra regressão linear porque mesmo que seu coeficiente seja alto, significando que a relação entre os dados
# é ok, vemos na reta da regressão que ele se deixa influenciar por um outliar e quase todos os seus valores são iguais

#o dataset R2 também não é adequado pra regressão linear pois vemos que a dispersão dos dados tem uma curvatura, ou seja, uma reta
# não representa bem esse conjunto de dados, de forma a não descrever corretamente a tendência desses dados
import matUtils

x1 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'x1')
y1 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'y1')
x2 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'x2')
y2 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'y2')
x3 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'x3')
y3 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'y3')


print(matUtils.regressao_multipla())

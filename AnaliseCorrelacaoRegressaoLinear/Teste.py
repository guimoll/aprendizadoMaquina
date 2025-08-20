import matUtils

x1 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'x1')
y1 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'y1')
print("R1: " + str(matUtils.regressao_linear(x1, y1)))

print("B1: " + str(matUtils.coeficiente_angular(x1, y1)))
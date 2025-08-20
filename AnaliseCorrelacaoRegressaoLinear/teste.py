# Guilherme Moll, Gabriel Ramos, Eduardo brandt

import matUtils

x1 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'x1')
y1 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'y1')
x2 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'x2')
y2 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'y2')
x3 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'x3')
y3 = matUtils.get_float_array_from_file(r'C:\Users\riley\Downloads\datasetFase1.txt', 'y3')


matUtils.representacao(x1, y1, "R1")
matUtils.representacao(x2, y2, "R2")
matUtils.representacao(x3, y3, "R3")


import matUtils
import graficos
# Guilherme Moll, Gabriel Ramos, Eduardo brandt


#=== ANÁLISE ESTATÍSTICA DOS DADOS ===
#              area    quartos          preco
#count    47.000000  47.000000      47.000000
#mean   2000.680851   3.170213  340412.765957
#std     794.702354   0.760982  125039.911223
#min     852.000000   1.000000  169900.000000
#25%    1432.000000   3.000000  249900.000000
#50%    1888.000000   3.000000  299900.000000
#75%    2269.000000   4.000000  384450.000000
#max    4478.000000   5.000000  699900.000000

#ITEM A: Média de preço das casas: $340,412.77
#ITEM B: Menor casa custa: $169,900.00
#   - Área: 1000.0 sq ft
#   - Quartos: 1.0
# Casa mais cara custa: $699,900.00
#   - Área: 4478.0 sq ft
#   - Quartos: 5.0

#ITEM C:
x,y = matUtils.load_data()
print(matUtils.load_data())

#ITEM D:
#x,y = matUtils.load_data_quartos_preco()
#demo.plot_dataset(x, y, "PLOT Numero de quartos e preço")
#z,v = matUtils.load_data_tamanho_casa_preco()
#demo.plot_dataset(z, v, "PLOT Tamanho casa e preço")

#ITEM E
#print(matUtils.regressao_multipla())
#b0,b1,b2 = matUtils.regressao_multipla()
#demo.plot_regression_3d(b0,b1,b2)
# Testar também com a escala normalizada (use_scaling=True)

#Item F,G
#X, Y = matUtils.load_data_tamanho_casa_preco()
#demo.plot_dataset_regressao_3d(X, Y, "PLOT 3D Tamanho casa e preço")
#Z,V = matUtils.load_data_quartos_preco()
#demo.plot_dataset_regressao_3d(Z, V, "Plot 3D Numero de quartos e preço")

#Item H
#b0, b1, b2 = matUtils.regressao_multipla()
#tamanho = 1650
#quartos = 3
#preco_previsto = b0 + b1 * tamanho + b2 * quartos
#print(f"Preço previsto para {tamanho} sqft e {quartos} quartos: {preco_previsto:.2f}")

#Item i
print("Preço vindo das libs:", matUtils.regressao_multipla_libs())
print("Preço vindo do método manual:", matUtils.regressao_multipla())